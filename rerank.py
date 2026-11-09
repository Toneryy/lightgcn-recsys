import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Dataset
from metrics import summary, top_k
from model import LightGCN

ROOT = Path(__file__).resolve().parent


def main():
    # штраф за популярность при ранжировании: score - alpha * log(1 + pop), модель не переобучается
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gowalla", "yelp2018"], required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.75,1")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Dataset(args.dataset)
    tag = f"{args.dataset}_k{args.layers}"
    model = LightGCN(data.n_users, data.n_items, data.norm_adj(device), layers=args.layers).to(device)
    model.emb.load_state_dict(torch.load(ROOT / "runs" / f"{tag}.pt", map_location=device))
    with torch.no_grad():
        u_emb, i_emb = model.propagate()
    log_pop = torch.log1p(torch.tensor(data.popularity, dtype=torch.float32, device=device))

    rows = []
    for alpha in map(float, args.alphas.split(",")):
        users, top = top_k(lambda u: u_emb[u] @ i_emb.T - alpha * log_pop, data, device=device)
        res = summary(users, top, data)
        res.pop("by_activity")
        rows.append({"alpha": alpha, **res})
        print(f"alpha {alpha}: recall {res['recall@20']:.4f} coverage {res['coverage']:.3f} "
              f"head {res['head_share_recs']:.3f} tail recall {res['tail_recall']:.4f}")
    (ROOT / "results" / f"{tag}_rerank.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
