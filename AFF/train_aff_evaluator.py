from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AFF.evaluator import AlphaForgeExpressionEvaluator
from AFF.zoo import build_default_expression_zoo
from calculator.calculator import InvalidEvaluateError, StockDataCalculator
from data.datatotensor import DataToTensorConfig, StockData
from expressions.expression import CLOSE, FORWARDRET


def _score_expression(calculator: StockDataCalculator, expr, metric: str) -> float:
    try:
        if metric == "ic":
            score = float(calculator.calc_single_IC(expr))
            return score if math.isfinite(score) else 0.0
        if metric == "rankic":
            score = float(calculator.calc_single_rankIC(expr))
            return score if math.isfinite(score) else 0.0
        raise ValueError(f"Unsupported metric: {metric}")
    except InvalidEvaluateError:
        return 0.0


def build_calculator(args) -> StockDataCalculator:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the AlphaForge-style evaluator on the current AlphaMiner expression zoo."
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stock-ids", nargs="*", default=None, help="Optional subset of stock ids.")
    parser.add_argument("--target-window", type=int, default=5)
    parser.add_argument("--metric", choices=["ic", "rankic"], default="ic")
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = build_calculator(args)
    exprs = build_default_expression_zoo()
    scores = [_score_expression(calculator, expr, args.metric) for expr in exprs]

    evaluator = AlphaForgeExpressionEvaluator(
        max_len=args.max_len,
        hidden=args.hidden,
        device=args.device,
    )
    fit_result = evaluator.fit(
        exprs,
        scores,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    pred = evaluator.predict(exprs).scores.detach().cpu()
    target = torch.tensor(scores, dtype=torch.float32)
    mse = torch.mean((pred - target) ** 2).item()
    corr = torch.corrcoef(torch.stack([pred, target]))[0, 1].item() if len(exprs) >= 2 else float("nan")

    print("device =", args.device)
    print("metric =", args.metric)
    print("zoo_size =", len(exprs))
    print("fit_result =", fit_result)
    print(f"train_mse = {mse:.6f}")
    print(f"train_corr = {corr:.6f}")
    print("--- top expressions by target score ---")
    ranked = sorted(zip(exprs, scores, pred.tolist()), key=lambda x: x[1], reverse=True)
    for expr, target_score, pred_score in ranked[:10]:
        print(f"{expr} | target={target_score:.6f} | pred={pred_score:.6f}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": evaluator.model.state_dict(),
                "config": {
                    "max_len": args.max_len,
                    "hidden": args.hidden,
                    "metric": args.metric,
                    "target_window": args.target_window,
                },
            },
            save_path,
        )
        print("saved_to =", save_path)


if __name__ == "__main__":
    main()
