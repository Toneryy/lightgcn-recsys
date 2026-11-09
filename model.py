import torch
import torch.nn.functional as F
from torch import nn


class SymSpMM(torch.autograd.Function):
    # матрица смежности симметрична, поэтому обратный проход – то же умножение
    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return adj @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.adj @ grad


class LightGCN(nn.Module):
    # layers=0 дает обычную матричную факторизацию (MF-BPR)
    def __init__(self, n_users, n_items, adj, dim=64, layers=3):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.adj, self.layers = adj, layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        e = total = self.emb.weight
        for _ in range(self.layers):
            e = SymSpMM.apply(self.adj, e)
            total = total + e
        total = total / (self.layers + 1)
        return total[:self.n_users], total[self.n_users:]

    def bpr_loss(self, users, pos, neg, decay):
        u_emb, i_emb = self.propagate()
        u, p, n = u_emb[users], i_emb[pos], i_emb[neg]
        loss = F.softplus((u * n).sum(1) - (u * p).sum(1)).mean()
        e0 = self.emb.weight
        reg = (e0[users].pow(2).sum() + e0[self.n_users + pos].pow(2).sum()
               + e0[self.n_users + neg].pow(2).sum()) / 2 / len(users)
        return loss + decay * reg, loss
