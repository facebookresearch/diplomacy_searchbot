# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional, Sequence
import os
import pathlib
import subprocess


OUTPUT_ROOT = pathlib.Path("/tmp/dip_tests")


def run_config(
    *, cfg: Union[pathlib.Path, str], overrides: Sequence[str], tag: Optional[str] = None
) -> None:
    """Runs the config with overrides and return path to workdir.

    Args:
        cfg: path to the config.
        overrides: list of overrides as passed on command line.
        tag: if provided, will save results to OUTPUT_ROOT/tag.

    """
    cmd = ["python", "run.py", "--cfg", str(cfg), "--mode", "restart", "--force"]
    if tag is not None:
        cmd.extend("--exp_id_pattern_override", tag)
    cmd.extend(overrides)

    print("Cmd:")
    print(f"HH_EXP_DIR={OUTPUT_ROOT}", *cmd)

    env = os.environ.copy()
    env["HH_EXP_DIR"] = str(OUTPUT_ROOT)
    if "USER" not in env:
        # Weird CI stuff.
        env["USER"] = "root"

    return subprocess.check_call(cmd, env=env)
