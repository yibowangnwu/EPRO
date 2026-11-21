#!/usr/bin/env python3
"""Apply the small EPRO integration to the pinned, clean verl checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(checkout, *args, check=True):
    result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result


def install(checkout, nvfp4=True):
    checkout = Path(checkout).resolve()
    lock = json.loads((ROOT / "configs/verl-lock.json").read_text())
    actual = git(checkout, "rev-parse", "HEAD").stdout.strip()
    if actual != lock["commit"]:
        raise RuntimeError(f"Expected verl {lock['commit']}; found {actual}. Use scripts/setup.sh.")
    destination = checkout / "verl/trainer/ppo/transported_sc.py"
    source = ROOT / "epro/transported_sc.py"
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise RuntimeError(f"Refusing to overwrite a different algorithm file: {destination}")
    patches = [ROOT / "patches/verl-transport.patch"]
    if nvfp4:
        patches.append(ROOT / "patches/verl-nvfp4-rollout.patch")
    # A repeat install is allowed only if every patch is already applied exactly.
    already = all(git(checkout, "apply", "--reverse", "--check", str(p), check=False).returncode == 0 for p in patches)
    if not already:
        for name, expected in lock["base_sha256"].items():
            got = hashlib.sha256((checkout / name).read_bytes()).hexdigest()
            if got != expected:
                raise RuntimeError(f"Locally modified or partially patched file: {name}. Use a clean checkout.")
        # Check the complete series before mutating the checkout.
        git(checkout, "apply", "--check", *map(str, patches))
        git(checkout, "apply", *map(str, patches))
    shutil.copyfile(source, destination)
    print(f"EPRO installed at {checkout} ({actual[:12]}; NVFP4 plumbing: {nvfp4})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--without-nvfp4", action="store_true")
    args = parser.parse_args()
    install(args.checkout, nvfp4=not args.without_nvfp4)
