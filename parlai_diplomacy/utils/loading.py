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
    import parlai_diplomacy.tasks.dialogue.agents  # noqa: F401
    import parlai_diplomacy.tasks.full_press.listener.single_order.regular.agents  # noqa: F401
    import parlai_diplomacy.tasks.full_press.listener.single_order.stream.agents  # noqa: F401
    import parlai_diplomacy.tasks.full_press.listener.all_orders.stream.agents  # noqa: F401
    import parlai_diplomacy.tasks.full_press.press.stream.agents  # noqa: F401
    import parlai_diplomacy.tasks.no_press.single_order.stream.agents  # noqa: F401
    import parlai_diplomacy.tasks.no_press.single_order.regular.agents  # noqa: F401
    import parlai_diplomacy.tasks.no_press.all_orders.stream.agents  # noqa: F401
    import parlai_diplomacy.tasks.new_reddit.agents  # noqa: F401

    # old data
    import parlai_diplomacy.tasks.no_press.single_order.legacy.agents  # noqa: F401


def register_all_scripts():
    # list all script modules here
    import parlai_diplomacy.scripts.display_game_order_preds  # noqa: F401
