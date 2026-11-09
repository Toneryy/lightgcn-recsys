import urllib.request
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta.*")

HF_URL = "https://huggingface.co/datasets/reczoo/{}/resolve/main/{}"
HF_NAMES = {"gowalla": "Gowalla_m1", "yelp2018": "Yelp18_m1"}
DATA_DIR = Path(__file__).resolve().parent / "data"


def download(name):
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for fname in ("train.txt", "test.txt"):
        path = folder / fname
        if not path.exists():
            print("скачиваю", HF_URL.format(HF_NAMES[name], fname))
            urllib.request.urlretrieve(HF_URL.format(HF_NAMES[name], fname), path)
    return folder


def read_pairs(path):
    users, items = [], []
    for line in path.read_text().splitlines():
        ids = [int(x) for x in line.split()]
        users += [ids[0]] * (len(ids) - 1)
        items += ids[1:]
    return np.array(users), np.array(items)


def to_csr(users, items, shape):
    m = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=shape)
    m.data[:] = 1
    return m


class Dataset:
    def __init__(self, name):
        folder = download(name)
        self.name = name
        tr_u, tr_i = read_pairs(folder / "train.txt")
        te_u, te_i = read_pairs(folder / "test.txt")
        self.n_users = int(max(tr_u.max(), te_u.max())) + 1
        self.n_items = int(max(tr_i.max(), te_i.max())) + 1
        shape = (self.n_users, self.n_items)
        self.train = to_csr(tr_u, tr_i, shape)
        self.test = to_csr(te_u, te_i, shape)
        self.popularity = np.asarray(self.train.sum(0)).ravel()
        self.user_degree = np.diff(self.train.indptr)
        self._keys = np.sort(tr_u.astype(np.int64) * self.n_items + tr_i)

    def stats(self):
        n_train, n_test = self.train.nnz, self.test.nnz
        return {"users": self.n_users, "items": self.n_items, "train": n_train, "test": n_test,
                "density": (n_train + n_test) / (self.n_users * self.n_items)}

    def sample(self, rng):
        # как в исходном коде LightGCN: пользователь равновероятно, затем его позитив и случайный негатив
        users = rng.choice(np.flatnonzero(self.user_degree), self.train.nnz)
        offs = (rng.random(len(users)) * self.user_degree[users]).astype(np.int64)
        pos = self.train.indices[self.train.indptr[users] + offs]
        neg = rng.integers(0, self.n_items, len(users))
        bad = self._is_train(users, neg)
        while bad.any():
            neg[bad] = rng.integers(0, self.n_items, bad.sum())
            bad[bad] = self._is_train(users[bad], neg[bad])
        return users, pos, neg

    def _is_train(self, users, items):
        keys = users.astype(np.int64) * self.n_items + items
        idx = np.searchsorted(self._keys, keys).clip(max=len(self._keys) - 1)
        return self._keys[idx] == keys

    def norm_adj(self, device):
        # симметрично нормированная матрица смежности графа пользователь-объект
        n = self.n_users + self.n_items
        r = self.train.tocoo()
        rows = np.concatenate([r.row, r.col + self.n_users])
        cols = np.concatenate([r.col + self.n_users, r.row])
        adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
        deg = np.asarray(adj.sum(1)).ravel()
        d = np.zeros_like(deg)
        d[deg > 0] = deg[deg > 0] ** -0.5
        adj = (sp.diags(d) @ adj @ sp.diags(d)).tocoo()
        idx = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        val = torch.tensor(adj.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (n, n), check_invariants=False).coalesce().to_sparse_csr().to(device)
