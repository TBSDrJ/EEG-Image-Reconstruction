"""
Author: Marcus Jaiclin (TBSDrJ), The Bishop's School
Date: Nov/Dec 2024
Program to detect and run all of the different variations.

This program only works if it is run from the parent folder of 
    ridge_comparisons, and all variations are named using format
    ridge_comparisons_{VERSION}.py.
"""
import pathlib
import subprocess

ran_something = True
while ran_something:
    ran_something = False
    base_path = pathlib.Path("ridge_comparisons")
    to_run = []
    for path in base_path.iterdir():
        if path.suffix != ".py":
            print(f"[INFO] runall.py skipping {path.name}")
            continue
        if "base" in path.name:
            print(f"[INFO] runall.py skipping {path.name}")
            continue
        if "run_all" in path.name:
            print(f"[INFO] runall.py skipping {path.name}")
            continue
        version = path.stem.split("_")[-1]
        saves_path = base_path / "saves"
        already_run = False
        for save_path in saves_path.iterdir():
            if save_path.suffix == ".txt":
                save_version = save_path.stem.split("_")[2]
                if save_version == version:
                    already_run = True
                    break
        if already_run:
            print(f"[INFO] runall.py skipping {path.name}")
            continue
        print(f"[INFO] runall.py starting {path.name}\n\n")
        subprocess.run(["python", path])
        ran_something = True
