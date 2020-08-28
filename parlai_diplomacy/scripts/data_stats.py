#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count and display statistics of the data.
Modified from parlai.script.data_stats
Examples
--------
.. code-block:: shell
For Bart:
  python scripts/data_stats.py -t state_order_chunk -dt valid:stream -ne 50000 --dict_tokenizer gpt2 --dict_file /private/home/wyshi/ParlAI/data/models/bart/bart_large/model.dict
For Transformer:
  python scripts/data_stats.py -t state_order_chunk -dt valid:stream -ne 50000 --dict-tokenizer bytelevelbpe --bpe-vocab /checkpoint/wyshi/diplomacy/context512_test_model/model.dict-vocab.json --bpe-merge /checkpoint/wyshi/diplomacy/context512_test_model/model.dict-merges.txt
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.data_stats import setup_args as setup_common_args

import parlai.utils.logging as logging
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()


def setup_args(parser=None):
    parser = setup_common_args(parser)
    parser.add_argument(
        "--input_truncate_size", type=int, default=1024, help="input sequence truncate length",
    )
    parser.add_argument(
        "--label_truncate_size", type=int, default=1024, help="label sequence truncate length",
    )
    parser.add_argument(
        "--extra_contents",
        type=str,
        default="input",
        help="input contents (state, state_history, order, dialogue etc), separated by comma, e.g. state,state_history",
    )
    parser.set_defaults(datatype="train:ordered")
    parser.set_defaults(dict_tokenizer="gpt2")
    parser.set_defaults(
        dict_file="/private/home/wyshi/ParlAI/data/models/bart/bart_large/model.dict"
    )
    DictionaryAgent.add_cmdline_args(parser)
    return parser


def report(world, counts, log_time, contents):
    report = world.report()
    stats = "\n"
    for t in contents + ["both"]:
        stats += t + ":\n"
        for s in [
            "utterances_in_",
            "avg_utterance_length_in_",
            "tokens_in_",
            "unique_tokens_in_",
            "unique_utterances_in_",
            "truncated_at_ratio_in_",
            "no_msg_ratio_in_",
        ]:
            snice = s.replace("_in_", "").replace("_", " ")
            if "truncated" in s:
                if t != "both":
                    for truncate_size in counts[s + t]:
                        stats += (
                            "   truncated at "
                            + str(truncate_size)
                            + ": "
                            + str(counts[s + t][truncate_size])
                            + "\n"
                        )
            elif "no_msg" in s:
                if t == "input":
                    stats += "   " + snice + ": " + str(counts[s + t]) + "\n"
            else:
                stats += "   " + snice + ": " + str(counts[s + t]) + "\n"
    log = {}
    log["stats"] = stats
    text, log = log_time.log(report["exs"], world.num_examples(), log)
    return text, log


def verify(opt, printargs=None, print_parser=None):
    if opt["datatype"] == "train":
        logging.warn("changing datatype from train to train:ordered")
        opt["datatype"] = "train:ordered"

    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    log_every_n_secs = opt.get("log_every_n_secs", -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float("inf")
    log_time = TimeLogger()

    dictionary = DictionaryAgent(opt)
    ignore_tokens = opt.get("ignore_tokens").split(",")

    # different truncate size
    text_truncate_sizes = [128, 256, 512, 1024, 2048]
    label_truncate_sizes = [128, 256, 512, 1024, 2048]
    if opt.get("input_truncate_size") not in text_truncate_sizes:
        text_truncate_sizes.append(opt.get("input_truncate_size"))
    if opt.get("label_truncate_size") not in label_truncate_sizes:
        label_truncate_sizes.append(opt.get("label_truncate_size"))

    contents = opt["extra_contents"].split(",")
    if "input" not in contents:
        contents = contents + ["input"]
    if "labels" not in contents:
        contents = contents + ["labels"]

    counts = {}
    for t in contents + ["both"]:
        counts["tokens_in_" + t] = 0
        counts["utterances_in_" + t] = 0
        counts["avg_utterance_length_in_" + t] = 0
        counts["unique_tokens_in_" + t] = 0
        counts["unique_utterances_in_" + t] = 0
        # for counting the stats..
        counts["token_dict_" + t] = {}
        counts["utterance_dict_" + t] = {}
        # for truncate size
        if t != "both":
            counts["truncated_at_in_" + t] = {
                truncate_size: 0 for truncate_size in text_truncate_sizes
            }
            counts["truncated_at_ratio_in_" + t] = {
                truncate_size: 0 for truncate_size in text_truncate_sizes
            }
    # how many datapoint has no msg
    counts["no_msg_in_input"] = 0
    counts["no_msg_ratio_in_input"] = 0

    def tokenize(txt):
        return dictionary.tokenize(txt)

    def keep_token(t):
        for s in ignore_tokens:
            if s != "" and s in t:
                return False
        return True

    # max number of examples to evaluate
    max_cnt = opt["num_examples"] if opt["num_examples"] > 0 else float("inf")
    cnt = 0

    # Show some example dialogs.
    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get("batchsize", 1)
        world.parley()
        act = world.get_acts()[opt.get("agent")]
        for itype in contents:
            if itype == "input":
                if opt.get("new_line_new_utt"):
                    txts = act.get("text").split("\n")
                else:
                    txts = [act.get("text")]

            elif itype == "labels":
                txts = act.get("labels", act.get("eval_labels", [""]))

            else:
                txts = act.get(itype)
                if type(txts) is not list:
                    txts = [txts]

            for txt in txts:
                # text sequence calculation
                tokens = tokenize(txt)
                retxt = []
                for t in tokens:
                    if keep_token(t):
                        retxt.append(t)
                counts["tokens_in_" + itype] += len(retxt)
                if itype in ["input", "labels"]:
                    counts["tokens_in_" + "both"] += len(retxt)
                counts["utterances_in_" + itype] += 1
                if itype in ["input", "labels"]:
                    counts["utterances_in_" + "both"] += 1
                counts["avg_utterance_length_in_" + itype] = (
                    counts["tokens_in_" + itype] / counts["utterances_in_" + itype]
                )
                if itype in ["input", "labels"]:
                    counts["avg_utterance_length_in_" + "both"] = (
                        counts["tokens_in_" + "both"] / counts["utterances_in_" + "both"]
                    )
                # truncate size calculation
                for truncate_size in counts["truncated_at_in_" + itype]:
                    if len(retxt) > truncate_size:
                        counts["truncated_at_in_" + itype][truncate_size] += 1
                for truncate_size in counts["truncated_at_in_" + itype]:
                    counts["truncated_at_ratio_in_" + itype][truncate_size] = (
                        counts["truncated_at_in_" + itype][truncate_size]
                        / counts["utterances_in_" + itype]
                    )

                # token-level calculation
                for t in retxt:
                    if t not in counts["token_dict_" + itype]:
                        counts["unique_tokens_in_" + itype] += 1
                        counts["token_dict_" + itype][t] = True
                    if itype in ["input", "labels"]:
                        if t not in counts["token_dict_" + "both"]:
                            counts["unique_tokens_in_" + "both"] += 1
                            counts["token_dict_" + "both"][t] = True
                retxt = " ".join(retxt)
                if retxt not in counts["utterance_dict_" + itype]:
                    counts["unique_utterances_in_" + itype] += 1
                    counts["utterance_dict_" + itype][retxt] = True
                if itype in ["input", "labels"]:
                    if retxt not in counts["utterance_dict_" + "both"]:
                        counts["unique_utterances_in_" + "both"] += 1
                        counts["utterance_dict_" + "both"][retxt] = True

        # how many data has no msg
        for itype in {"data_status"}:
            # data_status
            data_statuses = act.get("data_status")
            if itype == "data_status":
                data_statuses = act.get("data_status")

            if type(data_statuses) is not list:
                data_statuses = [data_statuses]

            # how many datapoint has no msg
            for data_status in data_statuses:
                if "NoMsg" in data_status:
                    counts["no_msg_in_input"] += 1
            counts["no_msg_ratio_in_input"] = (
                counts["no_msg_in_input"] / counts["utterances_in_" + "input"]
            )

        if log_time.time() > log_every_n_secs:
            text, log = report(world, counts, log_time, contents)
            if print_parser:
                logging.info(text)

    try:
        # print dataset size if available
        logging.info(
            f"loaded {world.num_episodes()} episodes with a total "
            f"of {world.num_examples()} examples"
        )
    except Exception:
        pass

    return report(world, counts, log_time, contents)


def obtain_stats(opt, parser):
    report_text, report_log = verify(opt, print_parser=parser)
    print(report_text.replace("\\n", "\n"))


@register_script("data_stats", hidden=True)
class DataStats(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return obtain_stats(self.opt, self.parser)


if __name__ == "__main__":
    DataStats.main()
