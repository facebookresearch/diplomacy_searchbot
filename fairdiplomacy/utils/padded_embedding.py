import torch


class PaddedEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        super().__init__()
        self.module = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=None if padding_idx < 0 else padding_idx,
            **kwargs
        )
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, indices):
        flat_indices = indices.view(-1)
        valid = (flat_indices != self.padding_idx).nonzero()  # nonzero() here is optional
        valid_indices = flat_indices[valid]
        valid_embs = self.module(valid_indices)

        flat_output = torch.zeros(
            flat_indices.shape[0],
            self.embedding_dim,
            dtype=valid_embs.dtype,
            device=valid_embs.device,
        )
        flat_output[valid] = valid_embs
        return flat_output.view(list(indices.shape) + [self.embedding_dim])
