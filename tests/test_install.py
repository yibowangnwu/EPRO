import os
from pathlib import Path
import subprocess

import pytest

from install_verl import install


@pytest.fixture
def clean_checkout(tmp_path):
    source = Path(os.environ.get("VERL_ROOT", "third_party/verl")).resolve()
    if not (source / "verl/trainer/config/ppo_trainer.yaml").is_file():
        pytest.skip("Set VERL_ROOT for patch application tests.")
    destination = tmp_path / "verl"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(source), str(destination)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "4b26c788d41ce932ce7706c5ee8e249ab52f4c51"],
                   check=True, capture_output=True)
    return destination


@pytest.mark.parametrize("nvfp4", [False, True])
def test_patches_apply_to_clean_checkout_and_repeat_unchanged(clean_checkout, nvfp4):
    install(clean_checkout, nvfp4=nvfp4)
    first = subprocess.check_output(["git", "-C", str(clean_checkout), "diff"])
    install(clean_checkout, nvfp4=nvfp4)
    assert subprocess.check_output(["git", "-C", str(clean_checkout), "diff"]) == first
    module = clean_checkout / "verl/trainer/ppo/transported_sc.py"
    compile(module.read_text(), str(module), "exec")


def test_modified_upstream_hook_is_rejected_before_changes(clean_checkout):
    path = clean_checkout / "verl/workers/utils/losses.py"
    changed = path.read_text() + "\n# A user's local change\n"
    path.write_text(changed)
    with pytest.raises(RuntimeError, match="Locally modified"):
        install(clean_checkout)
    assert path.read_text() == changed
    assert not (clean_checkout / "verl/trainer/ppo/transported_sc.py").exists()
