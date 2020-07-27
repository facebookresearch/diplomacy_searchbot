#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for loading agents and tasks, etc.
"""


def register_all_agents():
    # list all agents here
    import parlai_diplomacy.agents.special_tok_generator.agents  # noqa: F401


def register_all_tasks():
    # list all tasks here
    import parlai_diplomacy.tasks.language_diplomacy.agents  # noqa: F401
