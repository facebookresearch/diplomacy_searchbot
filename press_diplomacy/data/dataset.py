import copy
import glob
import torch
from abc import ABC, abstractmethod
from glob import glob

from parlai.core.agents import create_agent_from_model_file
from parlai.utils.torch import padded_tensor
from tqdm import tqdm

from fairdiplomacy.data.build_dataset import COUNTRY_ID_TO_POWER
from fairdiplomacy.data.dataset import *
from fairdiplomacy.models.dipnet.train_sl import get_sl_db_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)
logger.propagate = False


class PressDataset(Dataset, ABC):
    """
    Abstract Press Dataset class that every Press Dataset variant needs to subclass
    """

    def __init__(
        self,
        *,
        parlai_agent_file: str,
        message_chunks: str,
        game_ids: List[int],
        data_dir: str,
        game_metadata: Dict[int, Any],
        debug_only_opening_phase=False,
        only_with_min_final_score=7,
        n_jobs=20,
        value_decay_alpha=1.0,
        cf_agent=None,
        n_cf_agent_samples=1,
        min_rating=None,
        exclude_n_holds=-1,
    ):
        super().__init__(
            game_ids=game_ids,
            data_dir=data_dir,
            game_metadata=game_metadata,
            debug_only_opening_phase=debug_only_opening_phase,
            only_with_min_final_score=only_with_min_final_score,
            n_jobs=n_jobs,
            value_decay_alpha=value_decay_alpha,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            min_rating=min_rating,
            exclude_n_holds=exclude_n_holds,
        )

        message_files = glob(message_chunks)
        if len(message_files):
            self.message_files = message_files
        else:
            raise FileNotFoundError("Message chunk glob is empty.")

        # Due to the tokenizing overhead, num_jobs > 1 is not supported as it is a lot slower.
        if self.n_jobs != 1:
            logging.warning("num_dataloader_workers > 1 is not supported.")

        self.parlai_agent_file = parlai_agent_file
        self.dialogue_agent = create_agent_from_model_file(self.parlai_agent_file)
        self.default_tensor = self._encode_message(" ")
        self.messages, self.tokenized_messages = self._load_messages()

    @abstractmethod
    def _construct_message_dict(self, message_list: List[List[str]]) -> Dict:
        """
        Constructs message dict using raw message_list from the message db
        :param message_list: List[dict]
        :return:
        """
        raise NotImplementedError("Subclasses must implement this. ")

    @abstractmethod
    def _tokenize_message_dict(self) -> Dict:
        """
        Tokenizes messages in self.messages as constructed by _construct_message_dict
        :return:
        """
        raise NotImplementedError("Subclasses must implement this.")

    def _load_messages(self) -> Tuple[Dict, Dict]:
        """
        Loads message data
        """

        logging.info(f"Loading {len(self.message_files)} message chunks.")

        def load_message(message_json_path):
            try:
                with open(message_json_path, "r") as f:
                    database = json.load(f)
                    message_chunk = database[2]["data"]
            except FileNotFoundError:
                print(f"Error while loading message_chunk at {message_json_path}")
                return []
            else:
                return message_chunk

        messages_list = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(load_message)(file) for file in self.message_files
        )

        # TODO(apjacob): Perform caching to avoid loading each time
        # Flatten message chunks and constructs message_dict
        self.messages = self._construct_message_dict(
            [message for messages in messages_list for message in messages]
        )

        self.tokenized_messages = self._tokenize_message_dict()
        # Update game_ids
        self.game_ids = set(self.messages.keys()) & self.game_ids

        return self.messages, self.tokenized_messages

    def _encode_message(self, message: str) -> torch.Tensor:
        """
        Tokenizes message
        :param message: input message, str
        :return: Tensor
        """
        # TODO(apjacob): Currently accessing a protected member. Fix?
        assert (
            self.dialogue_agent is not None
        ), "Dialogue agent needs to be loaded to use the tokenizer"
        return self.dialogue_agent._vectorize_text(message, add_end=True)

    @abstractmethod
    def _encode_messages(self, game, game_id: int, phase_idx: int):
        """
        Encodes messages in game_id, phase_idx into a
        :param game: Diplomacy.Game object
        :param game_id: game id, int
        :param phase_idx: phase id, int
        :return: DataFields[TensorList]
        """
        raise NotImplementedError("Subclass must implement.")

    def _encode_phase(self, game, game_id: int, phase_idx: int, input_valid_power_idxs):
        data_fields = super(PressDataset, self)._encode_phase(
            game, game_id, phase_idx, input_valid_power_idxs
        )

        data_fields.update(self._encode_messages(game, game_id, phase_idx))

        return data_fields

    def validate_dataset(self):
        logging.info("Validating dataset..")
        for k, e in self.encoded_games.items():
            if isinstance(e, TensorList):
                if k == "x_input_message":
                    assert len(e) == self.num_phases * len(POWERS)
                else:
                    assert len(e) == self.num_phases * len(POWERS) * MAX_SEQ_LEN
            else:
                assert len(e) == self.num_phases


class ListenerDataset(PressDataset):
    def __init__(
        self,
        *,
        parlai_agent_file: str,
        message_chunks: str,
        game_ids: List[int],
        data_dir: str,
        game_metadata: Dict[int, Any],
        debug_only_opening_phase=False,
        only_with_min_final_score=7,
        n_jobs=20,
        value_decay_alpha=1.0,
        cf_agent=None,
        n_cf_agent_samples=1,
        min_rating=None,
        exclude_n_holds=-1,
    ):
        super().__init__(
            parlai_agent_file=parlai_agent_file,
            message_chunks=message_chunks,
            game_ids=game_ids,
            data_dir=data_dir,
            game_metadata=game_metadata,
            debug_only_opening_phase=debug_only_opening_phase,
            only_with_min_final_score=only_with_min_final_score,
            n_jobs=n_jobs,
            value_decay_alpha=value_decay_alpha,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            min_rating=min_rating,
            exclude_n_holds=exclude_n_holds,
        )

    def _construct_message_dict(self, message_list):
        message_dict = {}

        logging.info("Constructing message dict.")
        for message in tqdm(message_list):
            from_cnt_id = int(message["fromCountryID"])
            to_cnt_id = int(message["toCountryID"])

            if from_cnt_id not in COUNTRY_ID_TO_POWER or to_cnt_id not in COUNTRY_ID_TO_POWER:
                continue

            from_cnt = COUNTRY_ID_TO_POWER[from_cnt_id]
            to_cnt = COUNTRY_ID_TO_POWER[to_cnt_id]

            # select keys
            game_id = int(message["hashed_gameID"])
            phase_id = str(message["game_phase"])
            txt = str(message["message"])
            # set game dictionary
            message_dict.setdefault(game_id, {})
            # set turn dictionary
            message_dict[game_id].setdefault(phase_id, {})
            # isolate conversations between two players

            message_dict[game_id][phase_id].setdefault(to_cnt, [])

            # Only add listener messages
            message_dict[game_id][phase_id][to_cnt].append(txt)

        return message_dict

    def _tokenize_message_dict(self):
        assert self.messages is not None

        # Setting default tensor to be used later.
        self.tokenized_messages = copy.deepcopy(self.messages)
        logging.info("Tokenizing messages.")
        for game_id, phases in tqdm(self.messages.items()):
            for phase_id, countries in phases.items():
                for power_idx, power in enumerate(POWERS):
                    # Encodes for all powers regardless of whether they have messages
                    power_message_list = countries.get(power, [])
                    power_message = " ".join(power_message_list)
                    power_tensor = self._encode_message(power_message)
                    self.tokenized_messages[game_id][phase_id][power] = power_tensor

        return self.tokenized_messages

    # Note: HuggingFace Tokenizer cannot be pickled for joblib :(
    def _encode_messages(self, game, game_id: int, phase_idx: int) -> DataFields:
        assert game_id in self.tokenized_messages

        phase_name = game.get_phase_name(phase_idx)
        phase_tensor_dict = self.tokenized_messages[game_id].get(phase_name, dict())
        # TODO(apjacob): Verify country idx mapping
        # Construct the tensor in the order that the countries are listed in POWERS
        power_tensors = [
            phase_tensor_dict.get(power, self.default_tensor)
            for power_idx, power in enumerate(POWERS)
        ]
        padded_tensors, _ = padded_tensor(power_tensors, pad_idx=-1)
        input_message = TensorList.from_padded(padded_tensors, padding_value=-1)

        return DataFields(x_input_message=input_message)

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        assert self._preprocessed, "Dataset has not been pre-processed."

        fields = super().__getitem__(idx)

        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)

        assert isinstance(idx, torch.Tensor) and idx.dtype == torch.long
        assert idx.max() < len(self)

        idx //= self.n_cf_agent_samples
        x_idx = self.x_idxs[idx]
        power_idx = self.power_idxs[idx]

        x_input_message_idx = x_idx * len(POWERS) + power_idx
        x_input_message = self.encoded_games["x_input_message"][x_input_message_idx]
        fields["x_input_message"] = x_input_message.to_padded(padding_value=EOS_IDX).to(torch.long)

        return fields


def build_press_db_cache_from_cfg(cfg):
    game_metadata, min_rating, train_game_ids, val_game_ids = get_sl_db_args(
        cfg.metadata_path, cfg.min_rating_percentile, cfg.max_games, cfg.val_set_pct
    )

    train_dataset = ListenerDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        message_chunks=cfg.message_chunks,
        game_ids=train_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=cfg.num_dataloader_workers,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )
    train_dataset.preprocess()

    val_dataset = ListenerDataset(
        parlai_agent_file=cfg.parlai_agent_file,
        message_chunks=cfg.message_chunks,
        game_ids=val_game_ids,
        data_dir=cfg.data_dir,
        game_metadata=game_metadata,
        only_with_min_final_score=cfg.only_with_min_final_score,
        n_jobs=cfg.num_dataloader_workers,
        value_decay_alpha=cfg.value_decay_alpha,
        min_rating=min_rating,
        exclude_n_holds=cfg.exclude_n_holds,
    )
    val_dataset.preprocess()

    logger.info(f"Saving press datasets to {cfg.data_cache}")
    torch.save((train_dataset, val_dataset), cfg.data_cache)
