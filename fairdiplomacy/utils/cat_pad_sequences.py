import torch


def cat_pad_sequences(tensors, pad_value=0, pad_to_len=None, seq_dim=2, cat_dim=0):
    """
    Arguments:
    - tensors: a list of [B x 7 x S x ...] formatted tensors
    - pad_value: the value used to fill padding
    - pad_to_len: the desired total length. If none, use the longest sequence.

    Returns:
    - the result of torch.cat(tensors, dim=cat_dim) where each sequence has been
      padded to pad_to_len or the largest S
    - a list of S values for each tensor
    """
    seq_lens = [t.shape[seq_dim] for t in tensors]
    max_len = max(seq_lens) if pad_to_len is None else pad_to_len

    padded = [
        torch.cat(
            [
                t,
                torch.zeros(
                    *t.shape[:seq_dim],
                    max_len - t.shape[seq_dim],
                    *t.shape[seq_dim + 1 :],
                    dtype=t.dtype,
                    device=t.device,
                ).fill_(pad_value),
            ],
            dim=seq_dim,
        )
        if t.shape[seq_dim] < max_len
        else t
        for t in tensors
    ]
    return torch.cat(padded, dim=cat_dim), seq_lens
