from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main(args: Namespace) -> None:
    # df = pd.read_csv(args.csv)
    # print("length of entire dataframe:", len(df))

    data = []
    for p in tqdm(args.pickles):
        d = torch.load(p)
        print(f"length of {p}:", len(d))
        data.extend(d)

    # print("length of entire dataframe:", len(df))
    print("length of all data:", len(data))

    # mpid_to_results = {result['mp_id']: result for result in data}
    # ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
    #                    for idx in range(len(df))]
    torch.save(data, args.save)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("csv", type=Path, help="target csv containing cif files")
    parser.add_argument("save", type=Path, help="cache")
    parser.add_argument(
        "--pickles",
        metavar="pickles",
        required=True,
        type=Path,
        nargs="+",
        help="the pickles to combine",
    )
    args = parser.parse_args()

    main(args)
