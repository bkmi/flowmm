import os
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser
from tqdm import tqdm


def merge_data(sample_file: Path, initial_dataset_file: Path):
    targets = pd.read_csv(initial_dataset_file)
    targets["pretty_formula2"] = targets["cif"].apply(
        lambda x: Structure.from_str(x, fmt="cif").composition.formula
    )
    # print(targets.columns, targets.shape)

    samples = pd.read_csv(sample_file)
    samples.drop(columns=["gen_str", "model_name"], inplace=True)
    samples["pretty_formula2"] = samples["cif"].apply(
        lambda x: Structure.from_str(x, fmt="cif").composition.formula
    )
    # print(samples.columns, samples.shape)

    merged = samples.merge(targets, on="pretty_formula2", how="inner")
    # print(merged.head())
    # print(merged.columns, merged.shape)

    merged.drop(
        columns=[
            "pretty_formula_x",
            "pretty_formula2",
            "Unnamed: 0",
        ],
        inplace=True,
    )

    merged.rename(
        columns={
            "cif_x": "cif_initial",
            "pretty_formula_y": "pretty_formula",
            "cif_y": "cif",
        },
        inplace=True,
    )
    # print(merged.head())
    # print(merged.columns, merged.shape)
    return merged


def collate_data(dir: Path):
    def _get_formula(s):
        try:
            return AseAtomsAdaptor.get_atoms(
                CifParser.from_str(s).get_structures()[0]
            ).get_chemical_formula()
        except:
            return None

    merged_files = list(dir.glob("samples_*.csv"))
    print(f"Found {len(merged_files)} files")
    merged = pd.concat([pd.read_csv(f) for f in tqdm(merged_files)])
    if args.rename_to_init:
        # merged.rename(columns={"cif": "cif_initial"}, inplace=True)
        merged.drop(["gen_str", "model_name"], axis=1)
        merged["cif_initial"] = merged["cif"]
        merged["material_id"] = [f"material_{i}" for i in range(merged.shape[0])]
        merged["formation_energy_per_atom"] = [0 for i in range(merged.shape[0])]
        merged["band_gap"] = [0 for i in range(merged.shape[0])]
        # merged["pretty_formula"] = merged["cif"].apply(lambda s: AseAtomsAdaptor.get_atoms(CifParser.from_str(s).get_structures()[0]).get_chemical_formula())
        merged["pretty_formula"] = merged["cif"].apply(_get_formula)
        merged["e_above_hull"] = [0 for i in range(merged.shape[0])]
        merged["elements"] = [[] for i in range(merged.shape[0])]
        merged["spacegroup.number"] = [1 for i in range(merged.shape[0])]
    merged = merged[merged["pretty_formula"] != None]
    print(merged.columns, merged.shape, dir.with_suffix(".merged.csv"))
    # merged.to_csv(dir.with_suffix(".merged.csv"), index=False)
    merged.to_csv(dir.parent / f"{dir.name}.merged.csv", index=False)


def main(args):
    if args.collate_data:  # Collate all of the merged data
        collate_data(args.out_dir)
    else:  # Merge samples and targets
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            args.rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.rank is None:
            raise ValueError("Missing rank")

        out_file = args.out_dir / f"samples_{args.rank}.csv"
        if args.skip_existing and out_file.exists():
            print(f"Skipping existing {out_file}")
            return
        merged = merge_data(
            args.sample_dir / f"samples_{args.rank}.csv", args.initial_dataset_file
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_dataset_file", type=Path, default="data/basic.fixed/train.csv"
    )
    # parser.add_argument("--sample_dir", type=Path, default="exp/llama2-70B_lr0.0001_fp4_fixed/samples_0.9_0.99")
    # parser.add_argument("--out_dir", type=Path, default="exp/llama2-70B_lr0.0001_fp4_fixed/samples_0.9_0.99_merged/")
    # parser.add_argument("--sample_dir", type=Path, default="exp/gen_13b_t0.7_p0.99_train")
    # parser.add_argument("--out_dir", type=Path, default="exp/gen_13b_t0.7_p0.99_train_merged/")
    parser.add_argument(
        "--sample_dir",
        type=Path,
        default="exp/llama2-70B_lr0.0001_fp4_mp20alex40_r128_g8/checkpoint-165000/samples_train_t1.5",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="exp/llama2-70B_lr0.0001_fp4_mp20alex40_r128_g8/checkpoint-165000/samples_train_t1.5_merged",
    )
    parser.add_argument("--rank", type=Path, default=None)
    parser.add_argument("--collate_data", action="store_true")
    parser.add_argument("--rename_to_init", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    main(args)
