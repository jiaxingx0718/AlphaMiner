from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AFF.evaluator import AlphaForgeExpressionEvaluator
from AFF.generator import AlphaForgeGeneratorLSTM
from AFF.tokenizer import AlphaForgeTokenizer
from AFF.zoo import build_default_expression_zoo
from calculator.calculator import InvalidEvaluateError, StockDataCalculator
from data.datatotensor import DataToTensorConfig, StockData
from expressions.expression import CLOSE, FORWARDRET, Expression


@dataclass
class ZooItem:
    expr: Expression
    expr_str: str
    score: float
    source: str


def _build_calculator(args) -> StockDataCalculator:
    data_cfg = DataToTensorConfig(
        data_dir=PROJECT_ROOT / "data" / "cleaned" / "daily_cleaned",
        device=args.device,
    )
    data = StockData(
        config=data_cfg,
        selected_stock_ids=args.stock_ids if args.stock_ids else None,
    )
    return StockDataCalculator(
        data=data,
        target=FORWARDRET(CLOSE, args.target_window),
        max_invalid=1_000_000,
        cs_max_invalid=1_000_000,
        winsorize=True,
        normalize=True,
    )


def _score_expression(calculator: StockDataCalculator, expr: Expression, metric: str) -> float:
    try:
        if metric == "ic":
            score = float(calculator.calc_single_IC(expr))
        elif metric == "rankic":
            score = float(calculator.calc_single_rankIC(expr))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return score if math.isfinite(score) else 0.0
    except InvalidEvaluateError:
        return 0.0


def _build_initial_zoo(calculator: StockDataCalculator, metric: str) -> list[ZooItem]:
    items: list[ZooItem] = []
    for expr in build_default_expression_zoo():
        expr_str = str(expr)
        items.append(
            ZooItem(
                expr=expr,
                expr_str=expr_str,
                score=_score_expression(calculator, expr, metric),
                source="seed",
            )
        )
    return items


def _train_evaluator(
    zoo: list[ZooItem],
    *,
    max_len: int,
    hidden: int,
    device: str,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> tuple[AlphaForgeExpressionEvaluator, float]:
    evaluator = AlphaForgeExpressionEvaluator(max_len=max_len, hidden=hidden, device=device)
    fit_result = evaluator.fit(
        [item.expr for item in zoo],
        [item.score for item in zoo],
        lr=lr,
        batch_size=batch_size,
        num_epochs=epochs,
        patience=patience,
    )
    pred = evaluator.predict([item.expr for item in zoo]).scores.detach().cpu()
    target = torch.tensor([item.score for item in zoo], dtype=torch.float32)
    corr = torch.corrcoef(torch.stack([pred, target]))[0, 1].item() if len(zoo) >= 2 else float("nan")
    return evaluator, corr


def _train_generator_one_round(
    generator: AlphaForgeGeneratorLSTM,
    evaluator: AlphaForgeExpressionEvaluator,
    *,
    latent_size: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> float:
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    evaluator.model.eval()
    for p in evaluator.model.parameters():
        p.requires_grad_(False)

    last_pred_mean = float("nan")
    for _ in range(epochs):
        generator.train()
        z = torch.randn(batch_size, latent_size, device=device)
        out = generator.forward_masked_logits(z)
        onehot = F.gumbel_softmax(out.masked_logits, hard=True, dim=-1)
        pred = evaluator.model(onehot).squeeze(-1)
        loss = -pred.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_pred_mean = float(pred.mean().item())
    return last_pred_mean


def _sample_candidates(
    generator: AlphaForgeGeneratorLSTM,
    tokenizer: AlphaForgeTokenizer,
    calculator: StockDataCalculator,
    *,
    metric: str,
    latent_size: int,
    device: str,
    sample_size: int,
) -> tuple[list[ZooItem], dict[str, float]]:
    generator.eval()
    z = torch.randn(sample_size, latent_size, device=device)
    sample = generator.sample(z, deterministic=False)

    items: list[ZooItem] = []
    valid = 0
    for row in sample.action_ids.detach().cpu().tolist():
        try:
            expr = tokenizer.action_ids_to_expression(row)
            tokenizer.expression_to_action_ids(expr)
            expr_str = str(expr)
            score = _score_expression(calculator, expr, metric)
            items.append(ZooItem(expr=expr, expr_str=expr_str, score=score, source="generated"))
            valid += 1
        except Exception:
            continue

    stats = {
        "valid_rate": valid / max(sample_size, 1),
        "mean_score": (sum(item.score for item in items) / max(len(items), 1)) if items else 0.0,
        "best_score": max((item.score for item in items), default=float("-inf")),
    }
    return items, stats


def _refresh_zoo(
    zoo: list[ZooItem],
    new_items: list[ZooItem],
    *,
    max_size: int,
    keep_top_generated: int,
) -> list[ZooItem]:
    by_key: dict[str, ZooItem] = {}
    for item in zoo:
        prev = by_key.get(item.expr_str)
        if prev is None or item.score > prev.score:
            by_key[item.expr_str] = item

    ranked_new = sorted(new_items, key=lambda x: x.score, reverse=True)[:keep_top_generated]
    for item in ranked_new:
        prev = by_key.get(item.expr_str)
        if prev is None or item.score > prev.score:
            by_key[item.expr_str] = item

    merged = list(by_key.values())
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:max_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative AlphaForge-style trainer on top of AlphaMiner.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stock-ids", nargs="*", default=["000001", "000002", "000004", "000006", "000007"])
    parser.add_argument("--target-window", type=int, default=5)
    parser.add_argument("--metric", choices=["ic", "rankic"], default="ic")
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--hidden-p", type=int, default=128)
    parser.add_argument("--hidden-g", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--max-zoo-size", type=int, default=128)
    parser.add_argument("--keep-top-generated", type=int, default=24)
    parser.add_argument("--epochs-p", type=int, default=30)
    parser.add_argument("--patience-p", type=int, default=8)
    parser.add_argument("--batch-size-p", type=int, default=16)
    parser.add_argument("--lr-p", type=float, default=1e-3)
    parser.add_argument("--epochs-g", type=int, default=40)
    parser.add_argument("--batch-size-g", type=int, default=64)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = _build_calculator(args)
    tokenizer = AlphaForgeTokenizer(max_len=args.max_len)
    zoo = _build_initial_zoo(calculator, args.metric)

    generator = AlphaForgeGeneratorLSTM(
        latent_size=args.latent_size,
        d_model=args.hidden_g,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(args.device)
    generator.initialize_parameters()

    print("device =", args.device)
    print("metric =", args.metric)
    print("initial_zoo_size =", len(zoo))
    print("initial_best_score =", max(item.score for item in zoo))

    best_generated: list[ZooItem] = []

    for round_idx in range(args.rounds):
        evaluator, eval_corr = _train_evaluator(
            zoo,
            max_len=args.max_len,
            hidden=args.hidden_p,
            device=args.device,
            lr=args.lr_p,
            batch_size=args.batch_size_p,
            epochs=args.epochs_p,
            patience=args.patience_p,
        )
        pred_mean = _train_generator_one_round(
            generator,
            evaluator,
            latent_size=args.latent_size,
            device=args.device,
            epochs=args.epochs_g,
            batch_size=args.batch_size_g,
            lr=args.lr_g,
        )
        generated, stats = _sample_candidates(
            generator,
            tokenizer,
            calculator,
            metric=args.metric,
            latent_size=args.latent_size,
            device=args.device,
            sample_size=args.sample_size,
        )
        if generated:
            best_generated.extend(sorted(generated, key=lambda x: x.score, reverse=True)[: args.keep_top_generated])
            best_generated = sorted(best_generated, key=lambda x: x.score, reverse=True)[: args.max_zoo_size]
        zoo = _refresh_zoo(
            zoo,
            generated,
            max_size=args.max_zoo_size,
            keep_top_generated=args.keep_top_generated,
        )

        print(
            f"round={round_idx} "
            f"zoo_size={len(zoo)} "
            f"eval_corr={eval_corr:.4f} "
            f"gen_pred_mean={pred_mean:.4f} "
            f"valid_rate={stats['valid_rate']:.3f} "
            f"mean_true_{args.metric}={stats['mean_score']:.4f} "
            f"best_true_{args.metric}={stats['best_score']:.4f}"
        )

    print("--- top generated expressions across rounds ---")
    if best_generated:
        seen = set()
        top_unique = []
        for item in best_generated:
            if item.expr_str in seen:
                continue
            seen.add(item.expr_str)
            top_unique.append(item)
            if len(top_unique) >= 15:
                break
        for idx, item in enumerate(top_unique):
            print(f"[{idx}] score={item.score:.6f} | expr={item.expr_str}")
    else:
        print("no_valid_generated_expression")


if __name__ == "__main__":
    main()
