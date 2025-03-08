import multiprocessing
import multiprocessing.pool
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import torch

from flowmm.common.data_utils import preprocess_timeout


def main(args: Namespace) -> None:
    df = pd.read_csv(args.csv)
    print("length of entire dataframe:", len(df))
    print("start ind", args.start_ind)
    print("end ind", args.end_ind, "(could be None => until end)")
    df = df.iloc[args.start_ind : args.end_ind]
    print("length of dataframe portion", len(df))

    filtered_unordered_results = preprocess_timeout(
        df=df,
        num_workers=args.workers,
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
        prop_list=[args.prop],
        use_space_group=args.use_space_group,
        tol=args.tolerance,
        min_safe_crystal_volume=1,
        timeout=args.timeout,
    )
    torch.save(filtered_unordered_results, args.save)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv", type=Path, help="target csv containing cif files")
    parser.add_argument("save", type=Path, help="cache")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument(
        "--end_ind",
        type=int,
        default=None,
        help="default, None is until the end of the list",
    )
    parser.add_argument(
        "--prop",
        type=str,
        choices=["energy_per_atom", "formation_energy_per_atom", "heat_ref"],
        default="formation_energy_per_atom",
    )
    parser.add_argument("--niggli", type=bool, default=True)
    parser.add_argument("--primitive", type=bool, default=False)
    parser.add_argument(
        "--graph_method", type=str, choices=["crystalnn"], default="crystalnn"
    )
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--use_space_group", type=bool, default=False)
    parser.add_argument("--use_pos_index", type=bool, default=False)
    parser.add_argument(
        "--lattice_scale_method",
        type=str,
        choices=["scale_length"],
        default="scale_length",
    )
    args = parser.parse_args()

    main(args)
