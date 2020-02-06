import torch
import random


class CustomPackedSequence:
    def __init__(self, x, pad_val=0):
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.pad_val = pad_val

        flat = x.view(-1, x.shape[-1])
        self.lens = cand_lengths = x.shape[-1] - (flat == 0).sum(dim=1)
        self.data = flat.view(-1)[flat.view(-1) != pad_val]

    def unpack(self):
        x = torch.zeros(self.shape, dtype=self.dtype, device=self.device).fill_(self.pad_val)
        S = self.shape[-1]
        x = x.view(-1, S)
        c = 0
        for i, l in enumerate(self.lens):
            x[i, :l] = self.data[c : (c + l)]
            c += l
        return x.view(*self.shape)


if __name__ == "__main__":
    random.seed(0)
    B, P, S = 3, 7, 17
    x = torch.zeros(B, P, S, 469, dtype=torch.long)
    for a in range(B):
        for b in range(P):
            for c in range(S):
                x[a, b, c, : (0 if random.random() < 0.5 else random.randint(0, 20))] = 1

    p = CustomPackedSequence(x)
    print((p.unpack() == x).all())
