import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
import submitit
import torch
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
from torch_scatter import scatter

from fairchem.core import OCPCalculator
from fairchem.core.common.relaxation import OptimizableUnitCellBatch, ml_relax
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs

# for eSEN

#
# from fairchem.core import OCPCalculator
# OCPCalculator(
#     checkpoint_path="/checkpoint/xiangfu/emd_mptrj/checkpoints/eSEN_models/eSEN-30M-OMat/checkpoint.pt",
#     cpu=False,
#     seed=0,
# )


def get_primitive_cell(atoms: Atoms) -> Atoms:
    return AseAtomsAdaptor.get_atoms(
        AseAtomsAdaptor.get_structure(atoms).get_primitive_structure()
    )


def get_batch(
    df: pd.DataFrame,
    primitive: bool = False,
) -> Data:
    atoms_list = [
        AseAtomsAdaptor.get_atoms(Structure.from_str(s, fmt="cif")) for s in df["cif"]
    ]

    if primitive:
        atoms_list = [get_primitive_cell(a) for a in atoms_list]

    a2g = AtomsToGraphs(
        r_energy=False, r_forces=False, r_distances=False, r_pbc=True, r_edges=False
    )
    data_list = []
    for i, atoms in enumerate(atoms_list):
        data = a2g.convert(atoms)
        data.frac_coords = torch.tensor(atoms.get_scaled_positions()).float()
        data.sid = torch.tensor(i, dtype=torch.long)
        data_list.append(data)
    batch = data_list_collater(data_list, otf_graph=True)
    return batch


def wait_for_jobs_to_finish(jobs: list, sleep_time_s: int = 5) -> None:
    # wait for the job to be finished
    num_finished = 0
    print(f"number of jobs: {len(jobs):02d}")
    while num_finished < len(jobs):
        time.sleep(sleep_time_s)
        num_finished = sum(job.done() for job in jobs)
        print(f"number finished: {num_finished:02d}", flush=True, end="\r")
    print("")
    print("jobs done!")
    return None


def get_structures(batch: Data) -> list[Structure]:
    indexes = torch.cumsum(batch.natoms, dim=0)
    structures = []
    for i in range(len(batch.natoms)):
        start, end = 0 if i == 0 else indexes[i - 1], indexes[i]
        structures.append(
            Structure(
                lattice=batch.cell[i].cpu().numpy(),
                species=batch.atomic_numbers[start:end].cpu().numpy(),
                coords=batch.pos[start:end].cpu().numpy(),
                coords_are_cartesian=True,
                properties={
                    "sid": batch.sid[i].item(),
                    "converged": batch.converged[i].item(),
                    "energy": batch.energy[i].item(),
                },
            )
        )
    return structures


def relax(
    batch: OptimizableUnitCellBatch,
    checkpoint_path: Union[str, Path],
    fmax: float,
    maxiter: int,
    cpu: bool,
    seed: int = 0,
) -> OptimizableUnitCellBatch:
    # Download OMAT24 model from Hugging Face: https://huggingface.co/fairchem/OMAT24
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=cpu,
        seed=seed,
        trainer="equiformerv2_forces",
    )

    try:
        with torch.amp.autocast(
            device_type="cpu" if cpu else "cuda", enabled=calc.config["amp"]
        ):
            relaxed_batch = ml_relax(
                batch=batch,
                model=calc.trainer,
                steps=maxiter,
                fmax=fmax,
                relax_opt=None,
                relax_cell=True,
                relax_volume=False,
                save_full_traj=False,
                transform=None,
                mask_converged=True,
            )
    except RuntimeError:
        return None
    relaxed_batch.fmax = scatter(
        (relaxed_batch.forces**2).sum(axis=1).sqrt(),
        relaxed_batch.batch,
        reduce="max",
    )
    relaxed_batch.converged = relaxed_batch.fmax.ge(fmax)
    return relaxed_batch.to("cpu")


def main(args: Namespace) -> None:
    df = pd.read_csv(args.csv)
    print(f"{len(df)=}")
    num_structures = len(df)

    # limit num structures
    if args.num_structures is not None:
        assert args.num_structures <= num_structures
        num_structures = args.num_structures
    print(f"df max {num_structures=}")
    df = df.iloc[:num_structures, :]
    print(f"{len(df)=} after picking first {num_structures}")

    # get default out
    if args.out is None:
        args.out = args.csv.parent / "rfm_outputs_relaxed.csv"

    # split into chunks
    chunks = [
        df.iloc[i : i + args.batch_size, :] for i in range(0, len(df), args.batch_size)
    ]
    batches = [get_batch(chunk) for chunk in chunks]

    print(f"num batches: {len(batches)}")

    # setup cluster
    cluster = "local" if args.local else "slurm"
    if args.debug:
        cluster = "debug"
    executor = submitit.AutoExecutor(
        folder=args.csv.parent / "omatlog" if args.log_dir is None else args.log_dir,
        cluster=cluster,
    )
    executor.update_parameters(
        slurm_array_parallelism=args.num_jobs,
        nodes=1,
        slurm_ntasks_per_node=1,
        cpus_per_task=10,
        gpus_per_node=1,
        timeout_min=args.timeout_min,
        slurm_mem=30,
        slurm_account="ocp",
        slurm_qos="ami_shared",
    )
    doit = partial(
        relax,
        checkpoint_path=args.potential_path,
        # learning_rate=args.lr,
        fmax=args.fmax,
        maxiter=args.maxiter,
        cpu=args.cpu,
    )
    print("submitting work!")
    jobs = executor.map_array(doit, batches)

    if args.debug:
        for job in jobs:
            job.results()
    else:
        wait_for_jobs_to_finish(jobs)

    # process output
    relaxed_batches = [job.results()[0] for job in jobs if job.results()[0] is not None]
    if len(relaxed_batches) == 0:
        raise ValueError("no batches relaxed")
    else:
        converged = (
            torch.cat([rb.converged for rb in relaxed_batches]).sum().cpu().item()
        )
        print(f"num converged: {converged}")
        print(f"total: {num_structures}")

        structures = []
        for rb in relaxed_batches:
            structures.extend(get_structures(rb))

        df = pd.DataFrame(
            {
                "cif": [s.to(filename=None, fmt="cif") for s in structures],
                "converged": [s.properties["converged"] for s in structures],
                # "sid": [s.properties["sid"] for s in structures],
                "energy": [s.properties["energy"] for s in structures],
            }
        )
        print(f"saving ...")
        print(args.out)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "csv",
        type=Path,
        help="csv from `collected_to_cif.py`",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output csv, defaults to `rfm_outputs_relaxed.csv` in folder with input csv",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="number of jobs to divide structures between, only works when --slurm_partition is set",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--potential_path",
        default="/fsx-open-catalyst/bkmi/eqV2_86M_omat_mp_salex.pt",
        # default="/fsx-ocp-med/xiangfu/checkpoints/emd_mptrj/checkpoints/esen_omat_checkpoints/eSEN-30M-OAM-1plus8/checkpoint.pt",
        # default="/fsx-ocp-med/xiangfu/checkpoints/emd_mptrj/checkpoints/esen_omat_checkpoints/eSEN-30M-OMat/checkpoint.pt",
        choices=[
            "/fsx-ocp-med/xiangfu/checkpoints/emd_mptrj/checkpoints/esen_omat_checkpoints/eSEN-30M-OMat/checkpoint.pt",
            "/fsx-ocp-med/xiangfu/checkpoints/emd_mptrj/checkpoints/esen_omat_checkpoints/eSEN-30M-OAM-1plus8/checkpoint.pt",
            "/fsx-open-catalyst/bkmi/eqV2_86M_omat_mp_salex.pt",
        ],
        type=Path,
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--fmax",
        default=0.03,
        type=float,
    )
    parser.add_argument(
        "--maxiter",
        default=1_000,
        type=int,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu",
    )
    parser.add_argument("--timeout_min", type=int, default=180)
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the prerelaxations locally. overridden by debug",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run the prerelaxations sequentially in the same process as this script. overrides local",
    )
    args = parser.parse_args()

    main(args)
