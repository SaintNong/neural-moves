"""Train a move-ordering network on search-tree samples and export it.

    python -m nmo_train.train --train data/gen1/train --val data/gen1/validation \\
        --arch upn --out ../networks/upn.nmo

Loss (per node, over its non-hash moves with softmax probabilities p):
  efficiency  -log sum_m p_m r_m, where r_m is the move's first-pick efficiency
              (see data.py); the expected-efficiency objective.
  set         the same with r_m = 1 for every refutation (cost-agnostic).
  best        cross-entropy on the single most efficient move.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import Batch, Dataset, build_dataset, make_batch, padded
from .export import export_model
from .models import ARCHITECTURES

MVV_LVA = torch.tensor([[15, 14, 13, 12, 11, 10], [25, 24, 23, 22, 21, 20], [35, 34, 33, 32, 31, 30],
                        [45, 44, 43, 42, 41, 40], [55, 54, 53, 52, 51, 50], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]], dtype=torch.float32)


def mvv_lva_scores(batch: Batch) -> torch.Tensor:
    m = batch.moves
    table = MVV_LVA.to(m.device)
    capture = (m[:, 3] < 6).float()
    return capture * (1000 + table[m[:, 3], m[:, 2]]) + (m[:, 4] > 0).float() * 20000 + m[:, 4] * 100


def node_loss(batch: Batch, logits: torch.Tensor, loss: str) -> torch.Tensor:
    masked = torch.where(batch.valid, logits, torch.full_like(logits, -1e9))
    log_p = torch.log_softmax(padded(batch, masked, -1e9), dim=1)
    target = padded(batch, batch.target, 0.0)
    if loss == "best":
        return -log_p.gather(1, target.argmax(1, keepdim=True))[:, 0]
    log_r = torch.where(target > 0, target.clamp_min(1e-30).log(), torch.full_like(target, -1e9))
    return -torch.logsumexp(log_p + log_r, dim=1)


@torch.no_grad()
def ordering_cost(batch: Batch, scores: torch.Tensor) -> torch.Tensor:
    """Node cost of searching non-hash moves by descending score until the
    first refutation (stable: ties keep generator order). NaN for non-cut nodes."""
    masked = torch.where(batch.valid, scores, torch.full_like(scores, -1e30))
    order = torch.sort(padded(batch, masked, -1e31), dim=1, descending=True, stable=True).indices
    cost = padded(batch, batch.cost, 0.0).gather(1, order)
    cut = padded(batch, batch.cut & batch.valid, False).gather(1, order)
    first = cut.int().argmax(1, keepdim=True)
    total = cost.cumsum(1).gather(1, first)[:, 0]
    return torch.where(cut.any(1), total, torch.full_like(total, float("nan")))


@torch.no_grad()
def oracle_cost(batch: Batch) -> torch.Tensor:
    best = torch.where(batch.cut & batch.valid, batch.cost, torch.full_like(batch.cost, float("inf")))
    out = padded(batch, best, float("inf")).min(1).values
    return torch.where(torch.isinf(out), torch.full_like(out, float("nan")), out)


@torch.no_grad()
def evaluate(model, data: Dataset, loss: str, batch_size: int = 4096) -> dict:
    model.eval()
    totals = {}
    losses = []
    for start in range(0, len(data), batch_size):
        rows = torch.arange(start, min(start + batch_size, len(data)), device=data.depth.device)
        batch = make_batch(data, rows)
        logits = model(batch)
        losses.append(node_loss(batch, logits, loss))
        costs = {"model": ordering_cost(batch, logits), "mvv-lva": ordering_cost(batch, mvv_lva_scores(batch)),
                 "oracle": oracle_cost(batch)}
        depth = batch.depth.clamp(max=4)
        for name, cost in costs.items():
            for d in range(5):
                selected = (depth == d) & ~torch.isnan(cost)
                key = (name, d)
                totals[key] = totals.get(key, 0.0) + cost[selected].sum().item()
    model.train()
    report = {"loss": torch.cat(losses).mean().item()}
    for d in range(5):
        oracle = totals.get(("oracle", d), 0)
        if oracle > 0:
            report[f"d{d}"] = (totals[("model", d)] / oracle, totals[("mvv-lva", d)] / oracle)
    # Geometric mean over depths of model cost relative to MVV-LVA: the selection metric.
    ratios = [totals[("model", d)] / totals[("mvv-lva", d)] for d in range(5) if totals.get(("mvv-lva", d), 0) > 0]
    report["score"] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return report


def describe(report: dict) -> str:
    depths = " ".join(f"d{d} {report[f'd{d}'][0]:.2f}/{report[f'd{d}'][1]:.2f}" for d in range(5) if f"d{d}" in report)
    return f"loss {report['loss']:.4f} | cost/oracle model/mvv-lva: {depths} | rel {report['score']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, nargs="+")
    parser.add_argument("--val", required=True)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="upn")
    parser.add_argument("--model-args", default="{}", help="JSON keyword arguments for the architecture")
    parser.add_argument("--loss", choices=["efficiency", "set", "best"], default="efficiency")
    parser.add_argument("--tau", type=float, default=1.0, help="exponent on efficiencies")
    parser.add_argument("--no-failing-credit", action="store_true")
    parser.add_argument("--depth-weights", default="1,1,1,1,1", help="node weights for depths 0..4+")
    parser.add_argument("--cost-weight", type=float, default=0.0,
                        help="weight nodes by (cheapest refutation cost) ** this, emphasising costly decisions")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--table-lr-mult", type=float, default=1.0,
                        help="learning-rate multiplier for lookup tables (nn.Embedding), whose rows update sparsely")
    parser.add_argument("--max-train", type=int, default=0, help="cap on training nodes (0 for all)")
    parser.add_argument("--depths", default="", help="train and validate only on these remaining depths, e.g. 0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settings", default="{}", help="JSON settings stored in the .nmo file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = 0.0 if args.loss == "set" else args.tau
    start = time.time()
    depths = tuple(int(d) for d in args.depths.split(",")) if args.depths else None
    parts = [build_dataset(prefix, tau, not args.no_failing_credit, depths) for prefix in args.train]
    train = parts[0] if len(parts) == 1 else concat(parts)
    if args.max_train and len(train) > args.max_train:
        train = subset(train, args.max_train)
    val = build_dataset(args.val, tau, not args.no_failing_credit, depths)
    print(f"train {len(train)} nodes / {train.moves.shape[0]} moves, val {len(val)} nodes "
          f"({time.time() - start:.0f}s)", flush=True)
    train, val = train.to(device), val.to(device)

    model = ARCHITECTURES[args.arch](**json.loads(args.model_args)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.arch} {json.loads(args.model_args)}: {parameters} parameters", flush=True)
    depth_weights = torch.tensor([float(w) for w in args.depth_weights.split(",")], device=device)

    tables = [m.weight for m in model.modules() if type(m) is torch.nn.Embedding]
    others = [p for p in model.parameters() if all(p is not t for t in tables)]
    groups = [{"params": others, "lr": args.lr}, {"params": tables, "lr": args.lr * args.table_lr_mult}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                   pct_start=max(0.05, 3 / steps))
    best = None
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(train), device=device)
        running, count = 0.0, 0
        for begin in range(0, len(train), args.batch):
            batch = make_batch(train, permutation[begin:begin + args.batch])
            losses = node_loss(batch, model(batch), args.loss)
            weights = depth_weights[batch.depth.clamp(max=4)]
            if args.cost_weight:
                weights = weights * torch.nan_to_num(oracle_cost(batch), nan=1.0).pow(args.cost_weight)
            loss = (losses * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
            count += 1
        report = evaluate(model, val, args.loss)
        print(f"epoch {epoch + 1}: train {running / count:.4f} | {describe(report)} ({time.time() - start:.0f}s)", flush=True)
        if best is None or report["score"] < best:
            best = report["score"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            export_model(model.cpu(), args.out, **json.loads(args.settings))
            torch.save({"arch": args.arch, "model_args": json.loads(args.model_args), "state": model.state_dict()},
                       args.out.with_suffix(".pt"))
            model.to(device)
    print(f"best rel {best:.4f}; wrote {args.out}")


def concat(parts: list[Dataset]) -> Dataset:
    offsets, shift = [parts[0].offsets[:1]], 0
    for part in parts:
        offsets.append(part.offsets[1:] + shift)
        shift += part.moves.shape[0]
    fields = {name: torch.cat([getattr(p, name) for p in parts]) for name in Dataset.__dataclass_fields__ if name != "offsets"}
    return Dataset(offsets=torch.cat(offsets), **fields)


def subset(data: Dataset, count: int) -> Dataset:
    end = int(data.offsets[count])
    node_fields = ("features", "aux", "depth", "root")
    move_fields = ("moves", "target", "valid", "cost", "cut")
    return Dataset(offsets=data.offsets[:count + 1], **{f: getattr(data, f)[:count] for f in node_fields},
                   **{f: getattr(data, f)[:end] for f in move_fields})


if __name__ == "__main__":
    main()
