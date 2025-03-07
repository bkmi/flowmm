from argparse import ArgumentParser, Namespace
from pathlib import Path


def main(args: Namespace) -> None:
    checkpoints = list(args.root.glob("*/checkpoints/*.ckpt"))
    unique_runs = set([s.parents[1].stem for s in checkpoints])

    collector = {}
    # # this would load the newest epoch with zero padding...
    # checkpoints = sorted(checkpoints)
    # runs = {s.parents[1].stem: str(s.resolve()) for s in checkpoints}
    for ur in unique_runs:
        ckpts = [c for c in checkpoints if ur in str(c.parents[1].stem)]
        ckpts = sorted(ckpts, key=lambda c: int(c.stem.split("=")[1].split("-")[0]))
        collector[int(ur)] = ckpts[-1]

    for k in sorted(collector.keys()):
        print(collector[k])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    main(args)
