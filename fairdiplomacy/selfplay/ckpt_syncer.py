"""Simple mechanism to keep rollout models up to date.

CkptSyncer is initialized with a path to store checkpoints.

The trainer calls save() to save model in the folder.
Rollout worker class get_last_version() to get last version id and path to
the checkpoint. If the version differs from the last call, the worker should
reload the model.

The class can save/load arbitrary stuff. To work with torch modules check
helper functions save_state_dict and maybe_load_state_dict.

To prevent race conditions on NFS, the class maintains last `models_to_keep`
on the disk AND does atomic writes.
"""
from typing import List, Optional, Tuple
import glob
import logging
import os
import time

import torch


ModelVersion = int


class CkptSyncer:
    def __init__(self, prefix: str, models_to_keep: int = 10):
        self.prefix = prefix
        self.models_to_keep = models_to_keep

    def get_all_versions(self) -> List[Tuple[ModelVersion, str]]:
        versions = []
        for path in glob.glob(f"{self.prefix}_*"):
            if path.endswith(".tmp"):
                continue
            try:
                idx = int(path.split("_")[-1])
            except ValueError:
                logging.error("Bad file: %s", path)
                continue
            versions.append((idx, path))
        return sorted(versions)

    def save(self, obj) -> None:
        versions = self.get_all_versions()
        if versions:
            new_id = versions[-1][0] + 1
        else:
            new_id = 0
        path = f"{self.prefix}_{new_id:05d}"
        torch.save(obj, path + ".tmp")
        os.rename(path + ".tmp", path)
        models_to_delete = (len(versions) + 1) - self.models_to_keep
        if models_to_delete > 0:
            for _, path in versions[:models_to_delete]:
                os.remove(path)

    def get_last_version(self) -> Tuple[ModelVersion, str]:
        """Get last checkpoint and its version. Blocks if no checkpoints found."""
        while True:
            versions = self.get_all_versions()
            if not versions:
                logging.info("Waiting for checkpoint to appear (%s*)...", self.prefix)
                time.sleep(5)
                continue
            return versions[-1]

    def save_state_dict(self, torch_module) -> None:
        """Helper function to save model state."""
        return self.save(torch_module.state_dict())

    def maybe_load_state_dict(
        self, torch_module: torch.nn.Module, last_version: Optional[ModelVersion]
    ) -> ModelVersion:
        """Load model state if needed and return latest model version."""
        version, path = self.get_last_version()
        if version != last_version:
            torch_module.load_state_dict(path)
        return version
