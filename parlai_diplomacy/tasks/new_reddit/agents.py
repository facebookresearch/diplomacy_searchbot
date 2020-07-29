#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.loader import register_teacher
from parlai.core.teachers import FixedDialogTeacher, ChunkTeacher
from parlai.utils.safety import OffensiveLanguageClassifier
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.core.agents import create_agent_from_shared
from parlai.utils.misc import warn_once

from typing import Tuple, List

import gzip
import os
import logging
import random
import json
import torch

try:
    import sentencepiece as spm
except ImportError:
    print("sentencepiece is required to process data in meena way!")


FINAL_FILES_DIR = "/datasets01_101/reddit_2019/071919/chunks/"
PRESORTED_PATH = "/checkpoint/parlai/tasks/new_reddit_presorted/20190222_bpelower_ost+toronto+wiki"
MEENA_DATA_V1_PATH = "/checkpoint/parlai/tasks/meena_reddit/v1"
MEENA_DATA_V2_PATH = "/checkpoint/parlai/tasks/meena_reddit/v2"
MEENA_DATA_V2_SAFETY_PATH = "/checkpoint/parlai/tasks/meena_reddit/v2_safety_both"
BY_SUB_PATH = "/checkpoint/parlai/tasks/new_reddit/fifty_k_toks_per_sub_train.json"
CTRL_SUB_PATH = "/checkpoint/parlai/tasks/new_reddit/ctrl_sub_data/"
SENTENCEPIECE_PATH = "/private/home/daju/reddit_raw_sentences/model_shuffle_bpe/reddit_8k_default_model_shuf_bpe.model"
MOST_FREQUENT_SENTENCES_PATH = "/checkpoint/parlai/tasks/new_reddit_presorted_context_label_only/ulterences_more_than_100_sorted.txt"
NONENGLISH_SUBREDDITS = set(
    """
    mexico ateosmexicanos cinefilos explicacomoanino futbolmx futbol tigres
    rayados mejorde mexicocirclejerk mexicoiama mexijobs nahuatl nerdsmexico
    tvadicto videojuego gamingesp aguascalientes bajacalifornia tijuana
    mexicali ensenada rosarito distritofederal coyoacan aguascalientes
    bajacaliforniasur guanajuato estadodemexico oaxaca queretaro quintanaroo
    sinaloa sonora veracruz yucatan zacatecas edomex guadalajara guanajuato
    lalaguna monterrey playa es redditores argentina argentinacirclejerk
    argentos espanol bolivia chile santiago ecuador elsalvador latinoamerica
    paraguay peru colombia guatemala honduras nicaragua panama puertorico
    uruguay uruguaysoccer vzla arteperuano programacion djangoes role practicar
    madrid chistes noticias 15m valencia simpsonsespanollatino spanishimmersion
    france bretagne grandsud aixmarseille provence bordeaux montpellier
    toulouse lille lyon nantes fr cameroon morocco quebec montreal etsmtl udem
    uqam polymtl ulaval udes photosmontreal montrealgay montrealjobs
    montrealfood r4rmontreal occuponsmontreal occuponsmontrealag
    occuponsmontrealjobs spvm ecouteca entraideinformatique frenchrap jememarre
    jeuxvideo livres ligue1 programmation truefrance ragefrance lecafe
    demandezreddit frenchimmersion portuguese portugal porto aveiro coimbra
    lisboa desporto francesinhas benfica reddist brasil saopaulo riograndedosul
    brasilia bbrasil bossanova italy torino genova venezia napoli milano
    firenze italians erba soluzioni programmazione de deutschland frankfurt
    koeln germanpractice austria linz wien atomausstieg dokumentationen
    de_comedy germusic aachen germanmovies bern niedersachsen schleswigholstein
    hannover darmstadt bremen cologne hamburg frankfurt freiburg piratenpartei
    datenschutz de_it fragreddit teutonik germanyusa fernsehen hoerbuecher
    bundesliga heidelberg goe germantrees nederlands vlaanderen groningen
    nijmegen nl nederland antwerpen gent sweden svenska bostad jobb marknad
    swiama swadviceanimals ilandsproblem swedents swirclejerk metaswirclejerk
    stockholm malmo lund gothenburg uppsala linkoping umea skellefte gavle
    orebro vasteras karlstad boras halmstad denmark dkpol danishents dkmusik
    danskrap dkwork aalborg aarhus kolding odense norge oslo ntnu trondheim
    bergen stavanger polska polityka muzyka poznan lodz krakow ru
    russianimmersion suomi helsinki tampere chinesereddit cpop cflcomics
    japanesereddit nihongo nihon japaneseonly ja eli5
    """.strip().split()
)


def _uniqify(children):
    seen = set()
    for child in children:
        id_ = child["id"]
        if id_ in seen:
            continue
        seen.add(id_)
        yield child


def _find_ngrams(input_list, n):
    """
    Find ngrams of size n in input list.
    """
    return set(zip(*[input_list[i:] for i in range(n)]))


def annotate_safety(itr, batchsize=128, annotate="label"):
    """
    annotate safety on reddit data.

    Arguments:
        itr -- itr handle

    Keyword Arguments:
        batchsize {int} -- batch size (default: {128})
        annotate {str} -- choose between  (default: {'label', 'context', 'both'})

    Yields:
        item -- the annotated data
    """
    cls = OffensiveLanguageClassifier()
    cls_agent = cls.model
    cls_agent.model.eval()
    shared = cls_agent.share()
    copies = [create_agent_from_shared(shared) for _ in range(batchsize)]
    itr = iter(itr)
    batch = []
    while True:
        try:
            item = next(itr)
        except StopIteration:
            break

        batch.append(item)
        if len(batch) < batchsize:
            continue

        obs_context = []
        obs_label = []
        for c, item in zip(copies, batch):
            c.reset()
            obs_context.append(c.observe({"text": item["text"], "episode_done": True}))
            c.reset()
            obs_label.append(c.observe({"text": item["labels"][0], "episode_done": True}))

        if annotate != "both":
            obs = obs_label if annotate == "label" else obs_context
            with torch.no_grad():
                items_out = cls_agent.batch_act(obs)
            for pred_msg, item in zip(items_out, batch):
                _, _, pred, _, _, prob = pred_msg["text"].split()
                item[f"{annotate}_safety"] = pred == "__ok__"
                item[f"{annotate}_safety_prob"] = float(prob)
                yield item
        else:
            with torch.no_grad():
                items_out_context = cls_agent.batch_act(obs_context)
                items_out_label = cls_agent.batch_act(obs_label)
            for pred_msg_context, pred_msg_label, item in zip(
                items_out_context, items_out_label, batch
            ):
                _, _, pred_context, _, _, prob_context = pred_msg_context["text"].split()
                _, _, pred_label, _, _, prob_label = pred_msg_label["text"].split()
                item["context_safety"] = pred_context == "__ok__"
                item["context_safety_prob"] = float(prob_context)
                item["label_safety"] = pred_label == "__ok__"
                item["label_safety_prob"] = float(prob_label)
                yield item
        batch = []


class RedditChunksIterator:
    """
    Helper class that iterates through the chunks.

    Having it separate from the teacher makes it easier to deal with the multiple
    teacher instanciation done when batchsize > 1. Could potentially be made much faster
    by loading several chunks in parallel.
    """

    def __init__(
        self,
        chunks,
        length_cap=2048,
        is_train=True,
        english_only=False,
        subreddit_lst=None,
        exclude_subreddit_lst=None,
    ):
        self.original_chunks = chunks[:]
        self.length_cap = length_cap
        self.is_train = is_train
        self.english_only = english_only
        self.subreddit_lst = subreddit_lst
        self.exclude_subreddit_lst = exclude_subreddit_lst
        self.reset()

    def reset(self):
        self.chunks = self.original_chunks[:]
        self.current_chunk = self.chunks.pop(0)
        self.current_offset = 0
        self.samples = []
        self._load_samples(self.current_chunk)

    def recu_find_samples(
        self, element, text_history, collector, subreddit=None, outdomain=None, title=None,
    ):
        """
        Recursively find samples in the object element.
        """
        if self.subreddit_lst is not None and subreddit not in self.subreddit_lst:
            return
        if self.exclude_subreddit_lst is not None and subreddit in self.exclude_subreddit_lst:
            return

        if (
            "body" not in element
            or element["score"] is None
            or subreddit is None
            or element["body"] == "[deleted]"
            or element["body"] == "[removed]"
        ):
            return

        # drop everything with links
        body = element["body"]
        bodylower = body.lower()
        if "http://" in bodylower or "https://" in bodylower:
            return

        # unescape some common html tags
        body = body.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        body = body.replace("\n", " ").replace("\t", " ")
        body = body.strip()

        # if we've got a really excessive comment (~3 paragraphs by default),
        # then cut it off at the nearest space
        if len(body) > self.length_cap:
            try:
                body = body[: body.rindex(" ", 0, self.length_cap)]
            except ValueError:
                # we have a 2048 character utterance with zero spaces.
                # there's no way this is good english.
                return

        # drop really short messages
        if len(body) < 5:
            return

        # drop messages that don't start with a nice ascii, english character
        # strict, but it drops a lot of crap
        if ord(body[0]) > 127 or not body[0].isalpha():
            return

        if len(text_history) >= 1:
            collector.append(
                {
                    "text": "\n".join(text_history),
                    "labels": [body],
                    "episode_done": True,
                    "author_rank": element["author_rank"],
                    "author": element["author"],
                    "subreddit": subreddit,
                    "title": title,
                    "outdomain": outdomain,
                }
            )
        if "children" in element:
            text_history.append(body)
            for child in _uniqify(element["children"]):
                self.recu_find_samples(child, text_history, collector, subreddit, outdomain, title)
            text_history.pop()

    def _get_outdomain(self, domain):
        if domain.startswith("http://"):
            domain = domain[7:]
        if domain.startswith("https://"):
            domain = domain[8:]
        if domain.startswith("www."):
            domain = domain[4:]
        if "/" in domain:
            domain = domain[: domain.index("/")]
        return domain

    def _load_samples(self, chunk_id):
        """
        Load samples from chunk_id.
        """
        self.current_offset += len(self.samples)
        self.samples = []
        path = os.path.join(FINAL_FILES_DIR, "chunk_%04d.jsonl.gz" % chunk_id)
        logging.info("Loading samples from %s" % path)
        helper = []
        with gzip.open(path, "r") as f:
            for line in f:
                element = json.loads(line)
                # the first level is the submission
                sub = element.get("subreddit")
                outdomain = self._get_outdomain(element.get("url", ""))
                title = element.get("title")
                if "children" in element and len(element["children"]) > 0:
                    for child in _uniqify(element["children"]):
                        self.recu_find_samples(child, helper, self.samples, sub, outdomain, title)
        if self.is_train:
            # never be deterministic in training
            rand = random.Random()
        else:
            # always be deterministic in valid/test
            rand = random.Random(42)
        rand.shuffle(self.samples)
        logging.info("Loaded %d samples from %s" % (len(self.samples), path))

    def get(self, idx):
        """
        Return None if we're at the end of the dataset.
        """
        if self.is_train:
            if len(self.samples) == 0:
                self.chunks.append(self.current_chunk)
                self.current_chunk = self.chunks.pop(0)
                self._load_samples(self.current_chunk)
            return self.samples.pop(0)
        else:
            return self.samples[idx]


class RedditChunksSinglePassIterator(RedditChunksIterator):
    """
    Only passes through each chunk once.

    This is useful for performing data processing on each individual sample of the train
    set.
    """

    def get(self, idx):
        """
        Return None if we're at the end of the dataset.
        """
        offset_idx = idx - self.current_offset
        if offset_idx >= len(self.samples):
            if not self.is_train:
                return None
            if len(self.chunks) == 0:
                return None
            self.current_chunk = self.chunks.pop(0)
            self._load_samples(self.current_chunk)
            offset_idx = idx - self.current_offset
        return self.samples[offset_idx]


class RedditChunksMeenaIterator(RedditChunksIterator):
    """
    We also filter the data to improve the generation quality.

    A message is removed if any of the
    following conditions holds:
    1. the number of subwords is less than 2 or more than 128;
    2. the percentage of alphabetic characters is less than 70%;
    3. message contains URL;
    4. author’s username contains “bot”;
    5. the message is repeated more than 100 times;
    6. the message has a high n-gram overlap with the parent’s text;
    7. the message is potentially unsafe or offensive with respect to a commercial text classifier.
    In addition, we remove
    copies of the parent’s text quoted in a message.
    For simplicity, when a message is removed, we
    drop all sub-trees rooted under it. After these filtering steps, the number of (context, response)
    pairs extracted is 867M. The text is tokenized
    using byte-pair-encoding (BPE) (Sennrich et al.,
    2016) with the sentencepiece library.9 W
    """

    def __init__(
        self, chunks, length_cap=2048, is_train=True, english_only=True, subreddit_lst=None,
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(SENTENCEPIECE_PATH)
        with open(MOST_FREQUENT_SENTENCES_PATH) as f:
            content = f.readlines()
        self.most_frequent_ulterences = set([x.strip().strip('"') for x in content])
        super().__init__(chunks, length_cap, is_train, subreddit_lst=subreddit_lst)

    def recu_find_samples(
        self, element, text_history, collector, subreddit=None, outdomain=None, title=None,
    ):
        """
        Recursively find samples in the object element.
        """
        if self.subreddit_lst is not None and subreddit not in self.subreddit_lst:
            return

        if (
            "body" not in element
            or element["score"] is None
            or subreddit is None
            or element["body"] == "[deleted]"
            or element["body"] == "[removed]"
        ):
            return

        if self.english_only and subreddit.lower() in NONENGLISH_SUBREDDITS:
            return

        # drop everything with links
        body = element["body"]
        bodylower = body.lower()
        # drop everything appers more than 100 times.
        if body in self.most_frequent_ulterences:
            return

        if "bot" in element["author"].lower():
            return

        if "http://" in bodylower or "https://" in bodylower:
            return

        # unescape some common html tags
        body = body.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        body = body.replace("\n", " ").replace("\t", " ")
        body = body.strip()

        length = len(self.sp.EncodeAsIds(body))
        # drop everything has less than 2 subword or more than 128 subword
        if length < 2 or length > 128:
            return

        # drop everything with less than 70% alphabetic characters, space counts
        length_characters = float(len(body))
        filtered = [c for c in body if c.isalpha()]
        if float(len(filtered)) / length_characters < 0.7:
            return

        # compare n-gram overlap, drop those who has 80% overlaps
        if len(text_history) >= 1:
            parent_ngram = _find_ngrams(self.sp.EncodeAsIds(text_history[-1]), 3)
            body_ngram = _find_ngrams(self.sp.EncodeAsIds(body), 3)
            if len(body_ngram) > 0 and (len(parent_ngram & body_ngram) / len(body_ngram) >= 0.8):
                return

        if len(text_history) >= 1:
            collector.append(
                {
                    "text": "\n".join(text_history),
                    "labels": [body],
                    "episode_done": True,
                    "author_rank": element["author_rank"],
                    "title": title,
                    "author": element["author"],
                    "timestamp": int(element["created_utc"]),
                    "score": element["score"],
                    "gilded": element.get("gilded", 0),
                    "outdomain": outdomain,
                    "type": element["type"],
                    "subreddit": subreddit,
                    "id": element["id"],
                    "episode_done": True,
                }
            )
        if "children" in element:
            text_history.append(body)
            for child in _uniqify(element["children"]):
                self.recu_find_samples(child, text_history, collector, subreddit, outdomain, title)
            text_history.pop()


class NewRedditTeacher(FixedDialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument(
            "--subreddits",
            type=str,
            help="Which Subreddit(s). If not specified, keep all subreddits",
            default=None,
        )
        parser.add_argument(
            "--exclude-subreddits",
            type=str,
            help="Which Subreddits to exclude. If none, keep all subreddits.",
            default=None,
        )
        return parser

    def __init__(self, opt, shared=None):
        if opt.get("pytorch_teacher_task") is not None:
            raise ValueError(
                "NewRedditTeacher is not compatible with pytorch "
                "data teacher due to its large size. "
                "Found: pytorch_teacher_task=%s" % opt["pytorch_teacher_task"]
            )
        self.opt = opt
        self.is_train = False  # gets set in `_setup_chunk_nums`
        self._setup_chunk_nums(self.opt["datatype"])

        if opt["subreddits"] is not None:
            self.subreddit_lst = opt["subreddits"].split(",")
        else:
            self.subreddit_lst = None
        if opt["exclude_subreddits"] is not None:
            self.exclude_subreddit_lst = opt["exclude_subreddits"].split(",")
        else:
            self.exclude_subreddit_lst = None

        if shared is not None:
            self.iterator = shared["iterator"]
        else:
            chunks = list(range(self.start_chunk, self.end_chunk))
            if self.is_train and self.opt.get("distributed_world_size") is not None:
                dws = int(self.opt["distributed_world_size"])
                rank = int(self.opt["rank"])
                chunks = [c for c in chunks if c % dws == rank]

            # make sure if we're resuming from checkpoint, we'll get a completely
            # fresh random iteration
            if self.is_train:
                random.Random().shuffle(chunks)
            self.iterator = self._create_iterator(
                chunks,
                is_train=self.is_train,
                english_only=False,
                subreddit_lst=self.subreddit_lst,
            )
        super().__init__(opt, shared)
        self.reset()

    def _create_iterator(
        self, chunks, is_train, english_only=False, subreddit_lst=None,
    ):
        return RedditChunksIterator(
            chunks=chunks,
            is_train=is_train,
            english_only=english_only,
            subreddit_lst=self.subreddit_lst,
            exclude_subreddit_lst=self.exclude_subreddit_lst,
        )

    def reset(self):
        super().reset()
        self.iterator.reset()

    def _setup_chunk_nums(self, datatype):
        if "train" in datatype:
            self.is_train = True
            self.start_chunk = 0
            self.end_chunk = 4094
            self.num_samples = 2781060024
        elif "valid" in datatype:
            self.start_chunk = 4094
            self.end_chunk = 4095
            self.num_samples = 659055
        elif "test" in datatype:
            self.start_chunk = 4095
            self.end_chunk = 4096
            self.num_samples = 692865
        else:
            raise Exception("not a valid datatype: %s" % self.opt["datatype"])

    def num_episodes(self):
        return self.num_samples

    def num_examples(self):
        return self.num_samples

    def _setup_data(self, datatype):
        pass

    def get(self, episode_idx, entry_idx):
        return self.iterator.get(episode_idx)

    def share(self):
        shared = super().share()
        shared["iterator"] = self.iterator
        return shared


class MeenaRedditTeacher(NewRedditTeacher):
    def _create_iterator(self, chunks, is_train, english_only, subreddit_lst=None):
        # NOTE: English only should be set to True here
        return RedditChunksMeenaIterator(
            chunks=chunks, is_train=is_train, english_only=True, subreddit_lst=subreddit_lst,
        )


class SmallTeacher(NewRedditTeacher):
    def _setup_chunk_nums(self, datatype):
        if "train" in datatype:
            self.is_train = True
            self.start_chunk = 0
            self.end_chunk = 4094
            self.num_samples = 2781060024
        elif "valid" in datatype:
            self.start_chunk = 4094
            self.end_chunk = 4095
            self.num_samples = 10000
        elif "test" in datatype:
            self.start_chunk = 4095
            self.end_chunk = 4096
            self.num_samples = 10000
        else:
            raise Exception("not a valid datatype: %s" % datatype)


class NewRedditSinglePassTeacher(NewRedditTeacher):
    """
    Only passes through each chunk once.

    This is useful for performing data processing on each individual sample of the train
    set.
    """

    def __init__(self, opt, shared=None):
        if opt.get("pytorch_teacher_task") is not None:
            raise ValueError(
                "NewRedditTeacher is not compatible with pytorch "
                "data teacher due to its large size. "
                "Found: pytorch_teacher_task=%s" % opt["pytorch_teacher_task"]
            )
        self.opt = opt
        self.is_train = False
        self._setup_chunk_nums(self.opt["datatype"])
        if shared is not None:
            self.iterator = shared["iterator"]
        else:
            chunks = list(range(self.start_chunk, self.end_chunk))
            if self.is_train and self.opt.get("distributed_world_size") is not None:
                dws = int(self.opt["distributed_world_size"])
                rank = int(self.opt["rank"])
                chunks = [c for c in chunks if c % dws == rank]

            self.iterator = RedditChunksSinglePassIterator(chunks, is_train=self.is_train)
        super(NewRedditTeacher, self).__init__(opt, shared)
        self.reset()


class DefaultTeacher(NewRedditTeacher):
    pass


@register_teacher("presorted")
class PresortedTeacher(ChunkTeacher):
    data_delimiter = "\n"

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument("--delimiter", default="\n")
        return argparser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.delimiter = opt.get("delimiter")

    def _get_data_folder(self):
        return PRESORTED_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt["datatype"]
        if "train" in datatype:
            return 2213330368, 2213330368
        elif "valid" in datatype:
            return 613275, 613275
        elif "test" in datatype:
            return 601555, 601555

    def get_fold_chunks(self, opt) -> List[int]:
        datatype = opt["datatype"]
        if "train" in datatype:
            return list(range(4094))
        elif "valid" in datatype:
            return [4094]
        elif "test" in datatype:
            return [4095]

    def load_from_chunk(self, chunk_idx: int):
        """
        Refill the buffer.
        """
        path = os.path.join(self.folder, f"chunk{chunk_idx:04d}.txt.gz")

        output = []
        with gzip.open(path) as f:
            for line in f:
                line = line.decode("utf-8")
                if not line:
                    continue
                _, text, label = line.split("\t")
                label = label.rstrip()
                output.append((text, label))
        return output

    def create_message(self, sample_item, entry_idx=0):
        text, label = sample_item
        return {"text": text, "labels": [label], "episode_done": True}


@register_teacher("meena_data_v1")
class MeenaDataV1Teacher(PresortedTeacher):
    def _get_data_folder(self):
        return MEENA_DATA_V1_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt["datatype"]
        if "train" in datatype:
            # until 4093
            return 1086052180, 1086052180
        elif "valid" in datatype:
            # 4094
            return 268658, 268658
        elif "test" in datatype:
            # 4095
            return 261837, 261837

    def load_from_chunk(self, chunk_idx: int):
        """
        Refill the buffer.
        """
        path = os.path.join(self.folder, f"chunk{chunk_idx:04d}.txt.gz")

        output = []
        with gzip.open(path) as f:
            for line in f:
                line = line.decode("utf-8")
                if not line:
                    continue
                item = json.loads(line)
                text = item["context"]
                if self.delimiter is not None:
                    text = text.replace(self.data_delimiter, self.delimiter)
                label = item["label"]
                label = label.strip()
                output.append((text, label))
        return output


@register_teacher("meena_data_v2")
class MeenaDataV2Teacher(MeenaDataV1Teacher):
    def _get_data_folder(self):
        return MEENA_DATA_V2_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        # file lines count /checkpoint/parlai/tasks/meena_reddit/v2/sample_count_sorted.txt
        datatype = opt["datatype"]
        if "train" in datatype:
            # until 4093
            return 1187220396, 1187220396
        elif "valid" in datatype:
            # 4094
            return 294256, 294256
        elif "test" in datatype:
            # 4095
            return 285822, 285822


class MeenaDataV2SafetyTeacher(MeenaDataV1Teacher):
    def _get_data_folder(self):
        return MEENA_DATA_V2_SAFETY_PATH

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt["datatype"]
        # file lines count /checkpoint/parlai/tasks/meena_reddit/v2_safety_both/sample_count_sorted.txt
        if "train" in datatype:
            # until 4093
            return 1184731392, 1184731392
        elif "valid" in datatype:
            # 4094
            return 293632, 293632
        elif "test" in datatype:
            # 4095
            return 285184, 285184

    def load_from_chunk(self, chunk_idx: int):
        """
        Refill the buffer.
        """
        path = os.path.join(self.folder, f"chunk{chunk_idx:04d}.txt.gz")

        output = []
        with gzip.open(path) as f:
            for line in f:
                line = line.decode("utf-8")
                if not line:
                    continue
                item = json.loads(line)
                output.append(item)
        return output

    def create_message(self, sample_item, entry_idx=0):
        sample_item["episode_done"] = True
        return sample_item


class SafetyLabelsTeacher(MeenaDataV2SafetyTeacher):
    def create_message(self, sample_item, entry_idx=0):
        # randomly choose text or label
        text_vs_label = random.choice([0, 1])
        if text_vs_label:
            safety = sample_item["context_safety"]
            context = sample_item["text"]
        else:
            safety = sample_item["label_safety"]
            context = sample_item["labels"][0]

        new_ep = {
            "text": context,
            "labels": [OK_CLASS if safety else NOT_OK_CLASS],
            "episode_done": True,
            "id": "Reddit Safety Labels",
        }

        return new_ep


class SafetyLabelsSafeTeacher(SafetyLabelsTeacher):
    def load_from_chunk(self, chunk_idx: int):
        """
        Refill the buffer.
        """
        path = os.path.join(self.folder, f"chunk{chunk_idx:04d}.txt.gz")

        output = []
        with gzip.open(path) as f:
            for line in f:
                line = line.decode("utf-8")
                if not line:
                    continue
                item = json.loads(line)
                if item["label_safety"] or item["context_safety"]:
                    output.append(item)
        return output

    def create_message(self, sample_item, entry_idx=0):
        # randomly choose text or label
        if sample_item["context_safety"] and sample_item["label_safety"]:
            text_vs_label = random.choice([0, 1])
        elif sample_item["context_safety"]:
            text_vs_label = 1
        else:
            text_vs_label = 0

        if text_vs_label:
            safety = sample_item["context_safety"]
            context = sample_item["text"]
        else:
            safety = sample_item["label_safety"]
            context = sample_item["labels"][0]

        new_ep = {
            "text": context,
            "labels": [OK_CLASS if safety else NOT_OK_CLASS],
            "episode_done": True,
            "id": "Reddit Safety Labels",
        }

        return new_ep


class SafetyLabelsUnsafeTeacher(SafetyLabelsTeacher):
    def load_from_chunk(self, chunk_idx: int):
        """
        Refill the buffer.
        """
        path = os.path.join(self.folder, f"chunk{chunk_idx:04d}.txt.gz")

        output = []
        with gzip.open(path) as f:
            for line in f:
                line = line.decode("utf-8")
                if not line:
                    continue
                item = json.loads(line)
                if (not item["label_safety"]) or (not item["context_safety"]):
                    output.append(item)
        return output

    def create_message(self, sample_item, entry_idx=0):
        # randomly choose text or label
        if (not sample_item["context_safety"]) and (not sample_item["label_safety"]):
            text_vs_label = random.choice([0, 1])
        elif not sample_item["context_safety"]:
            text_vs_label = 1
        else:
            text_vs_label = 0
        if text_vs_label:
            safety = sample_item["context_safety"]
            context = sample_item["text"]
        else:
            safety = sample_item["label_safety"]
            context = sample_item["labels"][0]

        new_ep = {
            "text": context,
            "labels": [OK_CLASS if safety else NOT_OK_CLASS],
            "episode_done": True,
            "id": "Reddit Safety Labels",
        }

        return new_ep


class SubredditTeacher(FixedDialogTeacher):
    """
    Teacher where data stratified by subreddit.

    Contains ~3k subreddits, each with ~50k (BPE) tokens.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument(
            "--subreddits",
            type=str,
            help="Which Subreddit(s). If not specified, keep all subreddits",
            default=None,
        )
        parser.add_argument(
            "--reddit-datapath", type=str, help="Path to reddit data", default=BY_SUB_PATH,
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_path = opt.get("reddit_datapath", BY_SUB_PATH)
        self.id = "SubredditTeacher"
        self.subs = opt.get("subreddits").split(",")
        if shared:
            self.data = shared["data"]
        else:
            self._setup_data(opt.get("datatype").split(":")[0])

        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _setup_data(self, datatype):
        with open(self.data_path) as f:
            data = json.load(f)
        subreddits = set([d.lower() for d in data.keys()])
        self.data = []
        for sub in self.subs:
            if sub.lower() in subreddits:
                if sub in data:
                    self.data += data[sub]
                else:
                    warn_once(f"Missing subreddit: {sub}")
        if not self.data:
            self.data = [d for dd in data.values() for d in dd]

    def get(self, episode_idx, entry_idx):
        return self.data[episode_idx]

    def share(self):
        shared = super().share()
        shared["data"] = self.data
        return shared


class SubredditAllSubsTeacher(SubredditTeacher):
    """
    Controllable Teacher; call reset_data(sub) to reset data to that sub (prevents
    having to reload the data every time)
    """

    def _setup_data(self, datatype):
        with open(self.data_path) as f:
            self.full_data = json.load(f)
        self.possible_subs = list(self.full_data.keys())
        self.data = self.full_data[random.choice(self.possible_subs)]

    def reset_cur_data(self, sub=None):
        if sub is None:
            sub = random.choice(self.possible_subs)
        data = self.full_data[sub]
        texts = set()
        labels = set()
        self.data = []
        for d in data:
            if d["text"] in texts or d["labels"][0] in labels:
                texts.add(d["text"])
                labels.add(d["labels"][0])
            else:
                self.data.append(d)
                texts.add(d["text"])
                labels.add(d["labels"][0])


class SubredditCTRLSubsTeacher(SubredditTeacher):
    """
    Data from the subreddits described in the CTRL paper.

    Specify the `--subreddits` arg with comma-separated subreddits to use data from all
    those subs
    """

    @staticmethod
    def add_cmdline_args(parser):
        SubredditTeacher.add_cmdline_args(parser)
        parser.add_argument(
            "--num-exs-per-sub",
            type=int,
            default=-1,
            help="How many exs to include per sub. If <0, include all",
        )

    def __init__(self, opt, shared=None):
        opt["reddit_datapath"] = CTRL_SUB_PATH
        self.num_exs_per_sub = opt.get("num_exs_per_sub")
        super().__init__(opt, shared)

    def _setup_data(self, datatype):
        random.seed(42)
        self.data = []
        self.subs = [s.lower() for s in self.subs]
        data_files = os.listdir(self.data_path)
        subreddits_available = [d.replace(".json", "").lower() for d in data_files]
        subs_to_load = [
            data_files[i] for i, sub in enumerate(subreddits_available) if sub in self.subs
        ]
        if not subs_to_load:
            subs_to_load = ["full_data.json"]
        for sub in subs_to_load:
            with open(os.path.join(self.data_path, sub)) as f:
                data = json.load(f)
                random.shuffle(data)
                if "train" in datatype:
                    data = data[:-5000]
                    if self.num_exs_per_sub > 0 and len(data) > self.num_exs_per_sub:
                        data = data[: self.num_exs_per_sub]
                elif "valid" in datatype:
                    data = data[-5000:-2500]
                else:
                    data = data[-2500:]
                self.data += data
        if datatype == "train":
            random.shuffle(self.data)


class SubredditSplitByQualityTeacher(SubredditCTRLSubsTeacher):
    """
    Split/only serve data that is in the top X% of the data from a subreddit, based on
    perplexity.

    Specify a dir path that has files of the format `{subreddit}.json`, which map
    utterance idx to ppl, and keep X% of those utterances during training
    """

    @staticmethod
    def add_cmdline_args(parser):
        SubredditCTRLSubsTeacher.add_cmdline_args(parser)
        parser.add_argument(
            "--ppl-path", type=str, default=None, help="Dir in which ppl files exist"
        )
        parser.add_argument(
            "--top-ex-pct",
            type=float,
            default=0.5,
            help="percentage of top-performing (lowest ppl) exs to keep",
        )

    def __init__(self, opt, shared=None):
        self.ppl_path = opt["ppl_path"]
        self.top_ex_pct = opt["top_ex_pct"]
        if not os.path.isdir(self.ppl_path):
            raise RuntimeError("Must specify ppl path for this teacher")
        super().__init__(opt, shared)

    def _setup_data(self, datatype):
        self.data = []
        self.subs = [s.lower() for s in self.subs]
        data_files = os.listdir(self.data_path)
        subreddits_available = [d.replace(".json", "").lower() for d in data_files]
        subs_to_load = [
            data_files[i] for i, sub in enumerate(subreddits_available) if sub in self.subs
        ]
        if not subs_to_load:
            subs_to_load = ["full_data.json"]
        for sub in subs_to_load:
            random.seed(42)
            with open(os.path.join(self.data_path, sub)) as f:
                data = json.load(f)
            with open(os.path.join(self.ppl_path, sub)) as f:
                ppl_utts = json.load(f)

            sorted_ppl_utts = [int(i[0]) for i in sorted(ppl_utts.items(), key=lambda x: x[1])]
            data = list(enumerate(data))
            random.shuffle(data)
            idx_to_new_idx, data = zip(*data)
            idx_to_new_idx = {j: i for i, j in enumerate(idx_to_new_idx)}
            sorted_ppl_utts = [idx_to_new_idx[idx] for idx in sorted_ppl_utts]
            if "train" in datatype:
                data = data[:-5000]
                sorted_ppl_utts = [s for s in sorted_ppl_utts if s < len(data)]
                if self.num_exs_per_sub > 0 and len(data) > self.num_exs_per_sub:
                    data = data[: self.num_exs_per_sub]
                    sorted_ppl_utts = [s for s in sorted_ppl_utts if s < self.num_exs_per_sub]
            elif "valid" in datatype:
                data = data[-5000:-2500]
            else:
                data = data[-2500:]
            if "train" in datatype:
                num_exs_to_keep = int(self.top_ex_pct * len(data))
                data = [data[idx] for idx in sorted_ppl_utts[:num_exs_to_keep]]
            self.data += data
        if datatype == "train":
            random.shuffle(self.data)
