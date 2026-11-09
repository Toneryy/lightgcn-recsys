import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import Dataset
from metrics import summary, top_k
from model import LightGCN

ROOT = Path(__file__).resolve().parent


def run_popular(data, args, device):
    pop = torch.tensor(data.popularity, dtype=torch.float32, device=device)
    users, top = top_k(lambda u: pop.expand(len(u), -1).clone(), data, args.k, device=device)
    return {"history": [], "final": summary(users, top, data)}


def run_lightgcn(data, args, device):
    model = LightGCN(data.n_users, data.n_items, data.norm_adj(device), args.dim, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device == "cuda")
    rng = np.random.default_rng(args.seed)
    history = []

    def evaluate():
        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model.propagate()
        users, top = top_k(lambda u: u_emb[u] @ i_emb.T, data, args.k, device=device)
        model.train()
        return users, top

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        users, pos, neg = (torch.tensor(x, device=device) for x in data.sample(rng))
        perm = torch.randperm(len(users), device=device)
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            loss, bpr = model.bpr_loss(users[b], pos[b], neg[b], args.decay)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += bpr.item() * len(b)
        row = {"epoch": epoch, "loss": total / len(perm), "sec": time.time() - t0}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # тест только для кривой обучения, итог берется с последней эпохи
            res = summary(*evaluate(), data)
            row.update({m: res[m] for m in (f"recall@{args.k}", f"ndcg@{args.k}")})
            print(f"{data.name} K={args.layers} epoch {epoch}: loss {row['loss']:.4f} "
                  f"recall {row[f'recall@{args.k}']:.4f} ndcg {row[f'ndcg@{args.k}']:.4f}", flush=True)
        history.append(row)

    users, top = evaluate()
    if args.save_runs:
        out = ROOT / "runs"
        out.mkdir(exist_ok=True)
        torch.save(model.emb.state_dict(), out / f"{args.tag}.pt")
        np.save(out / f"{args.tag}_top.npy", top)
    return {"history": history, "final": summary(users, top, data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gowalla", "yelp2018"], required=True)
    ap.add_argument("--model", choices=["lightgcn", "pop"], default="lightgcn")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--save-runs", action="store_true")
    args = ap.parse_args()
    args.tag = f"{args.dataset}_pop" if args.model == "pop" else f"{args.dataset}_k{args.layers}"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Dataset(args.dataset)
    t0 = time.time()
    res = run_popular(data, args, device) if args.model == "pop" else run_lightgcn(data, args, device)
    res.update({"config": {k: v for k, v in vars(args).items() if k != "save_runs"},
                "data": data.stats(), "minutes": (time.time() - t0) / 60, "device": torch.cuda.get_device_name()
                if device == "cuda" else "cpu"})
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    f = res["final"]
    print(f"{args.tag}: recall@{args.k} {f[f'recall@{args.k}']:.4f} ndcg@{args.k} {f[f'ndcg@{args.k}']:.4f}")


if __name__ == "__main__":
    main()
