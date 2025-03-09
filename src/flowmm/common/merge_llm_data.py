from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main(args: Namespace) -> None:
    print("number of csvs:", len(args.csvs))

    collector = []
    for csv in tqdm(args.csvs):
        df = pd.read_csv(csv)
        collector.append(df)
    df = pd.concat(collector)

    if args.source is not None:
        source = pd.read_csv(args.source)
        raise NotImplementedError("must implement join")

    print("length of entire dataframe:", len(df))
    df.to_csv(args.save)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("save", type=Path, help="file to save join and merged dfs")
    parser.add_argument(
        "--csvs",
        metavar="csvs",
        required=True,
        type=Path,
        nargs="+",
        help="the csvs to combine",
    )
    parser.add_argument(
        "--source", type=Path, default=None, help="train/val/test.csv to join on"
    )
    args = parser.parse_args()

    main(args)
