import numpy as np
import torch


@torch.no_grad()
def top_k(score_fn, data, k=20, batch=2048, device="cuda"):
    # полное ранжирование по всем объектам, просмотренные в train исключаются
    users = np.flatnonzero(np.diff(data.test.indptr))
    top = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        scores = score_fn(torch.tensor(u, device=device))
        seen = data.train[u].tocoo()
        scores[torch.tensor(seen.row, device=device), torch.tensor(seen.col, device=device)] = -torch.inf
        top[s:s + len(u)] = scores.topk(k).indices.cpu().numpy()
    return users, top


def hits_matrix(users, top, data):
    keys = np.sort(np.repeat(np.arange(data.n_users), np.diff(data.test.indptr)).astype(np.int64)
                   * data.n_items + data.test.indices)
    q = users[:, None].astype(np.int64) * data.n_items + top
    idx = np.searchsorted(keys, q).clip(max=len(keys) - 1)
    return keys[idx] == q


def accuracy(users, top, data):
    k = top.shape[1]
    hits = hits_matrix(users, top, data)
    n_test = np.diff(data.test.indptr)[users]
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = np.cumsum(disc)[np.minimum(n_test, k) - 1]
    return {"recall": hits.sum(1) / n_test, "ndcg": (hits * disc).sum(1) / idcg,
            "precision": hits.sum(1) / k}


def summary(users, top, data):
    k = top.shape[1]
    per_user = accuracy(users, top, data)
    res = {f"{m}@{k}": float(v.mean()) for m, v in per_user.items()}

    # смещение к популярному: покрытие каталога, Gini, доля «головы» в выдаче
    counts = np.bincount(top.ravel(), minlength=data.n_items)
    srt = np.sort(counts)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ srt / (n * srt.sum()))
    order = np.argsort(-data.popularity)
    head = np.zeros(data.n_items, dtype=bool)
    head[order[:int(0.2 * data.n_items)]] = True
    test_items = data.test.indices
    hits = hits_matrix(users, top, data)
    tail_hits = (hits & ~head[top]).sum()
    res.update({
        "coverage": float((counts > 0).mean()),
        "gini": gini,
        "head_share_recs": float(head[top].mean()),
        "head_share_train": float(data.popularity[head].sum() / data.popularity.sum()),
        "head_share_test": float(head[test_items].mean()),
        "avg_pop_recs": float(data.popularity[top].mean()),
        "avg_pop_test": float(data.popularity[test_items].mean()),
        "tail_recall": float(tail_hits / (~head[test_items]).sum()),
        "head_recall": float((hits & head[top]).sum() / head[test_items].sum()),
    })

    # качество по группам пользователей: квартили по числу взаимодействий в train
    deg = data.user_degree[users]
    edges = np.quantile(deg, [0, 0.25, 0.5, 0.75, 1])
    groups = np.clip(np.searchsorted(edges, deg, side="right") - 1, 0, 3)
    res["by_activity"] = [{"group": g + 1, "min": int(deg[groups == g].min()), "max": int(deg[groups == g].max()),
                           "users": int((groups == g).sum()),
                           f"recall@{k}": float(per_user["recall"][groups == g].mean()),
                           f"ndcg@{k}": float(per_user["ndcg"][groups == g].mean())} for g in range(4)]
    return res
