# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from typing import Callable, Dict, List, Optional, Tuple
import glob
import logging
import os
import pathlib
import time

from conf import agents_cfgs
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent

import torch


ModelVersion = int


class CkptSyncer:
    def __init__(self, prefix: str, models_to_keep: int = 10, create_dir=False, ckpt_extra=None):
        self.prefix = prefix
        self.models_to_keep = models_to_keep
        if create_dir:
            pathlib.Path(prefix).parent.mkdir(exist_ok=True, parents=True)
        self._last_loaded_model_meta = {}

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
        path = f"{self.prefix}_{new_id:08d}"
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

    def save_state_dict(self, torch_module, args=None, **meta) -> None:
        """Helper function to save model state."""
        torch_module = getattr(torch_module, "module", torch_module)
        state = {"model": torch_module.state_dict(), "meta": meta, "args": args}
        return self.save(state)

    def maybe_load_state_dict(
        self, torch_module: torch.nn.Module, last_version: Optional[ModelVersion]
    ) -> ModelVersion:
        """Load model state if needed and return latest model version."""
        version, path = self.get_last_version()
        if version != last_version:
            pickle = torch.load(path, map_location="cpu")
            torch_module.load_state_dict(pickle["model"])
            self._last_loaded_model_meta = pickle.get("meta", {})
        return version

    def get_meta(self) -> Dict:
        return self._last_loaded_model_meta


class ValuePolicyCkptSyncer:
    """A holder for 2 separate syncers for policy and value models."""

    def __init__(self, prefix: str, models_to_keep: int = 10, create_dir=False, ckpt_extra=None):
        prefix = prefix.strip(".")
        kwargs = dict(models_to_keep=models_to_keep, create_dir=create_dir, ckpt_extra=ckpt_extra)
        self.value = CkptSyncer(f"{prefix}.value", **kwargs)
        self.policy = CkptSyncer(f"{prefix}.policy", **kwargs)

    def items(self):
        return dict(value=self.value, policy=self.policy).items()


def build_search_agent_with_syncs(
    searchbot_cfg: agents_cfgs.SearchBotAgent,
    *,
    ckpt_sync_path: Optional[str],
    use_trained_policy: bool,
    use_trained_value: bool,
    device_id: Optional[int] = None,
) -> Tuple[SearchBotAgent, Callable[[], Dict[str, Dict]]]:
    """Builds a search agent using some of ckpts from the syncer and a reload function.

    Performs the following modifications on the agent config before loading it:
        * model_path: use one from the syncer if use_trained_policy is set
        * value_model_path: use one from the syncer if use_trained_value is set
        * device: set to device_id if provided.

    If ckpt_sync_path is None, loads the agent with default params. In this
    case use_trained_policy and use_trained_value must be False.

    Returns tuple of 2 elements:
        searchbot_agent: the agent.
        do_sync_fn: on call loads new weights from checkpoints into the agent
            and returns a dict: syncer -> meta.

    Meta is a dict with meta information about the ckeckpoint as provided by
    trainer during syncer.save_state_dict call.
    """
    searchbot_cfg = searchbot_cfg.to_editable()
    if ckpt_sync_path is not None:
        logging.info("build_search_agent_with_syncs: Waiting for ckpt syncers")
        assert use_trained_policy or use_trained_value
        joined_ckpt_syncer = ValuePolicyCkptSyncer(ckpt_sync_path)
        # If using trained policy and/or value, need to get their paths so that we
        # can construct a proper model.
        _, last_policy_path = joined_ckpt_syncer.policy.get_last_version()
        _, last_value_path = joined_ckpt_syncer.value.get_last_version()
        logging.info("build_search_agent_with_syncs: Original agent_one cfg:\n%s", searchbot_cfg)
        default_model_path = searchbot_cfg.model_path
        default_value_model_path = searchbot_cfg.value_model_path or default_model_path
        searchbot_cfg.model_path = last_policy_path if use_trained_policy else default_model_path
        searchbot_cfg.value_model_path = (
            last_value_path if use_trained_value else default_value_model_path
        )
    else:
        logging.info("build_search_agent_with_syncs: dummy call with no syncers")
        assert not use_trained_policy
        assert not use_trained_value
    if device_id is not None:
        searchbot_cfg.device = device_id
    searchbot_cfg = searchbot_cfg.to_frozen()
    logging.info(
        "build_search_agent_with_syncs: The following agent_one cfg will be used:\n%s", searchbot_cfg
    )
    # FIXME(akhti): I'm not sure it's needed.
    skip_model_cache = not use_trained_value or not use_trained_value
    agent = SearchBotAgent(searchbot_cfg, skip_model_cache=skip_model_cache)

    sync_tuples = []
    versions = {}
    if use_trained_value:
        sync_tuples.append(("value", joined_ckpt_syncer.value, agent.model.value_model))
        versions["value"] = None
    if use_trained_policy:
        sync_tuples.append(("policy", joined_ckpt_syncer.policy, agent.model.model))
        versions["policy"] = None

    def do_sync():
        metas = {}
        for name, syncer, model in sync_tuples:
            versions[name] = syncer.maybe_load_state_dict(model, versions[name])
            metas[name] = syncer.get_meta()
        return metas

    return (agent, do_sync)
