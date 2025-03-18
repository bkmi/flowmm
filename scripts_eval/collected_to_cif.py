from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

from flowmm.pymatgen_ import diffcsp_to_structure


def main(args: Namespace) -> None:
    graphs = {k: v[0] for k, v in torch.load(args.consolidated).items()}
    num_structures = graphs["num_atoms"].squeeze().numel()

    collector = []
    for i in trange(num_structures):
        structure = diffcsp_to_structure(graphs, i, clip_atom_types=False)
        collector.append(structure.to(fmt="cif"))

    df = pd.DataFrame(collector, columns=["cif"])

    if args.out is None:
        args.out = args.consolidated.parent / "rfm_outputs.csv"
    df.to_csv(args.out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("consolidated", type=Path, help="consolidated_rfm-from-llm.pt")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output csv, defaults to `rfm_outputs.csv` in folder",
    )
    parser.add_argument(
        "--cif_prefix", type=str, default="mat_", help="cif prefix names"
    )
    args = parser.parse_args()

    main(args)
