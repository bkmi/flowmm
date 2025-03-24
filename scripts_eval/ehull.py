from __future__ import annotations

import pickle
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

PATH_PPD_MP = Path(__file__).parents[1] / "mp_02072023/2023-02-07-ppd-mp.pkl"


def generate_MP_compatible_CSE(
    structure: Structure, total_energy: float
) -> ComputedStructureEntry:
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        assert len(poscar.site_symbols) == len(
            incar["LDAUU"]
        ), f"Number of LDAUU values ({len(incar['LDAUU'])}) does not match the number of site symbols in the POSCAR ({len(poscar.site_symbols)})"
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": total_energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    MaterialsProject2020Compatibility(
        check_potcar=False,
    ).process_entries(
        cse, clean=True, inplace=True
    )  # noqa: PD002, RUF100

    # Return the final CSE (notice that the composition/etc is also clean, not things like Fe3+)!
    return cse


def main(args: Namespace) -> None:
    with open(PATH_PPD_MP, "rb") as f:
        ppd_mp = pickle.load(f)
    df = pd.read_csv(args.csv)
    df["energy_uncorrected"] = df["energy"]

    collector = {
        "num sites": [],
        "composition": [],
        "correction": [],
        "energy": [],
        "energy_per_atom": [],
        "hull_energy_per_atom": [],
        "energy_above_hull_per_atom": [],
    }
    for cif, energy_uncorrected in zip(
        df["cif"].to_list(), df["energy_uncorrected"].to_list()
    ):
        structure = Structure.from_str(cif, fmt="cif")
        cse = generate_MP_compatible_CSE(structure, energy_uncorrected)
        collector["num sites"].append(structure.num_sites)
        collector["composition"].append(cse.composition.to_pretty_string())
        collector["correction"].append(cse.correction)
        collector["energy"].append(cse.energy)
        collector["energy_per_atom"].append(cse.energy_per_atom)
        try:
            hull_energy_per_atom = ppd_mp.get_hull_energy_per_atom(
                structure.composition
            )
        except (ValueError, AttributeError, ZeroDivisionError):
            hull_energy_per_atom = float("nan")
        collector["hull_energy_per_atom"].append(hull_energy_per_atom)
        collector["energy_above_hull_per_atom"].append(
            cse.energy_per_atom - hull_energy_per_atom
        )

    for key, value in collector.items():
        df[key] = value

    df.to_csv(args.csv, index=False)
    print(f"wrote file to: ")
    print(f"{args.csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv", type=Path, help="prerelaxed dataframe")
    # parser.add_argument("json_out", type=Path, help="new dataframe")
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--clean_outputs_dir",
        type=Path,
        default=None,
        help="root dir for vasp clean_outputs",
    )
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument("--method", type=str, default=None)

    args = parser.parse_args()

    main(args)
