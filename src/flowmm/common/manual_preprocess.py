import multiprocessing
import multiprocessing.pool
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from tqdm import trange

from flowmm.common.data_utils import process_one


def process_one_star(args):
    return process_one(*args)


def main(args: Namespace) -> None:
    df = pd.read_csv(args.csv)
    print("length of entire dataframe:", len(df))
    print("start ind", args.start_ind)
    print("end ind", args.end_ind, "(could be None => until end)")
    df = df.iloc[args.start_ind : args.end_ind]
    print("length of dataframe portion", len(df))

    with multiprocessing.Pool(processes=args.workers) as pool:
        argss = [
            (*elm,)
            for elm in zip(
                [df.iloc[idx] for idx in range(len(df))],
                [args.niggli] * len(df),
                [args.primitive] * len(df),
                [args.graph_method] * len(df),
                [[args.prop]] * len(df),  # the list is actually nested!
                [args.use_space_group] * len(df),
                [args.tolerance] * len(df),
            )
        ]
        unordered_results = []
        work = pool.imap_unordered(func=process_one_star, iterable=argss)
        for _ in trange(len(argss)):
            try:
                unordered_results.append(work.next(timeout=args.timeout))
            except multiprocessing.TimeoutError:
                continue
    # unordered_results = p_umap(
    #     process_one,
    #     [df.iloc[idx] for idx in range(len(df))],
    #     [args.niggli] * len(df),
    #     [args.primitive] * len(df),
    #     [args.graph_method] * len(df),
    #     [[args.prop]] * len(df),  # the list is really nested
    #     [args.use_space_group] * len(df),
    #     [args.tolerance] * len(df),
    #     num_cpus=args.workers,
    # )

    # filter bad cifs
    filtered_unordered_results = []
    for ur in unordered_results:
        if ur is None:
            continue
        elif "graph_arrays_initial" in ur.keys() and ur["graph_arrays_initial"] is None:
            continue
        else:
            filtered_unordered_results.append(ur)
    torch.save(filtered_unordered_results, args.save)
    # mpid_to_results = {result['mp_id']: result for result in filtered_unordered_results}
    # ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
    #                    for idx in range(len(df))]
    # torch.save(ordered_results, args.save)


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
