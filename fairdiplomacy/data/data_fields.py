# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairdiplomacy.utils.tensorlist import TensorList


class DataFields(dict):
    BOOL_STORAGE_FIELDS = ["x_board_state", "x_prev_state"]

    def select(self, idx):
        return DataFields({k: v[idx] for k, v in self.items()})

    @classmethod
    def cat(cls, L: list):
        if len(L) > 0:
            return cls({k: _cat([x[k] for x in L]) for k in L[0]})
        else:
            return cls()

    @classmethod
    def stack(cls, L: list, dim: int = 0):
        if len(L) > 0:
            return cls({k: torch.stack([x[k] for x in L], dim) for k in L[0]})
        else:
            return cls()

    def to_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.bool)
        return self

    def from_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        return DataFields(
            {k: v.to(*args, **kwargs) if hasattr(v, "to") else v for k, v in self.items()}
        )


def _cat(x):
    return TensorList.cat(x) if isinstance(x[0], TensorList) else torch.cat(x)
