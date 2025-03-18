import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import submitit
import torch
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_scatter import scatter

from fairchem.core import OCPCalculator
from fairchem.core.common.relaxation import OptimizableUnitCellBatch, ml_relax
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs


def get_primitive_cell(atoms: Atoms) -> Atoms:
    return AseAtomsAdaptor.get_atoms(
        AseAtomsAdaptor.get_structure(atoms).get_primitive_structure()
    )


def get_batch(
    df: pd.DataFrame,
    primitive: bool = False,
):
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


def relax(
    batch: OptimizableUnitCellBatch,
    checkpoint_path: Union[str, Path],
    fmax: float,
    maxiter: int,
    cpu: bool,
    seed: int = 42,
) -> OptimizableUnitCellBatch:
    # Download OMAT24 model from Hugging Face: https://huggingface.co/fairchem/OMAT24
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=cpu,
        seed=seed,
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
    num_structures = len(df)

    # limit num structures
    if args.num_structures is not None:
        assert args.num_structures <= num_structures
        num_structures = args.num_structures
    print(f"{num_structures=}")
    df = df.iloc[:num_structures, :]

    # get default out
    if args.out is None:
        args.out = args.csv.parent / "rfm_outputs_relaxed.pt"

    # split into chunks
    chunks = [
        df.iloc[i : i + args.batch_size, :] for i in range(0, len(df), args.batch_size)
    ]
    batches = [get_batch(chunk) for chunk in chunks]

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
        checkpoint_path=args.omat_path,
        # learning_rate=args.lr,
        fmax=args.fmax,
        maxiter=args.maxiter,
        cpu=args.cpu,
    )
    print("submitting work!")
    jobs = executor.map_array(doit, batches)

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
        with open(args.out.parent / f"{args.out.stem}_converged.txt", "w") as f:
            f.write(f"{converged} / {num_structures}")
        torch.save(relaxed_batches, args.out)


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
        help="output pickle, defaults to `rfm_outputs_relaxed.pt` in folder with csv",
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
        default=32,
    )
    parser.add_argument(
        "--omat_path",
        default="/fsx-open-catalyst/bkmi/eqV2_86M_omat_mp_salex.pt",
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
        default=2_000,
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
