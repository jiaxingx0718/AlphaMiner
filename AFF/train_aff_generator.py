from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from dataclasses import dataclass

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
from expressions.expression import CLOSE, FORWARDRET


@dataclass
class DecodedSample:
    expr_str: str | None
    score: float | None
    valid: bool


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


def _score_expression(calculator: StockDataCalculator, expr, metric: str) -> float:
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


def _train_evaluator(args, calculator: StockDataCalculator) -> AlphaForgeExpressionEvaluator:
    exprs = build_default_expression_zoo()
    scores = [_score_expression(calculator, expr, args.metric) for expr in exprs]
    evaluator = AlphaForgeExpressionEvaluator(
        max_len=args.max_len,
        hidden=args.hidden_p,
        device=args.device,
    )
    evaluator.fit(
        exprs,
        scores,
        lr=args.lr_p,
        batch_size=args.batch_size_p,
        num_epochs=args.epochs_p,
        patience=args.patience_p,
    )
    evaluator.model.eval()
    for p in evaluator.model.parameters():
        p.requires_grad_(False)
    return evaluator


def _loss_simi(onehot_1: torch.Tensor, onehot_2: torch.Tensor, threshold: float) -> torch.Tensor:
    simi = torch.sum(onehot_1 * onehot_2, dim=-1).sum(dim=-1)
    simi = simi / onehot_1.shape[1]
    return torch.relu(simi - threshold).mean()


def _loss_potential(latent_1: torch.Tensor, latent_2: torch.Tensor, threshold: float) -> torch.Tensor:
    eps = 1e-7
    u1 = latent_1.clamp(eps, 1 - eps)
    u2 = latent_2.clamp(eps, 1 - eps)
    sim = (u1 * u2).sum(dim=1) / (((u1 ** 2).sum(dim=1).sqrt()) * ((u2 ** 2).sum(dim=1).sqrt()))
    return torch.relu(sim - threshold).mean()


def _decode_samples(
    tokenizer: AlphaForgeTokenizer,
    action_ids: torch.Tensor,
    calculator: StockDataCalculator,
    metric: str,
    limit: int = 16,
) -> list[DecodedSample]:
    rows = action_ids.detach().cpu().tolist()[:limit]
    results: list[DecodedSample] = []
    for row in rows:
        try:
            expr = tokenizer.action_ids_to_expression(row)
            score = _score_expression(calculator, expr, metric)
            results.append(DecodedSample(expr_str=str(expr), score=score, valid=True))
        except Exception:
            results.append(DecodedSample(expr_str=None, score=None, valid=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal AlphaForge-style generator training on top of AlphaMiner.")
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
    parser.add_argument("--epochs-p", type=int, default=40)
    parser.add_argument("--patience-p", type=int, default=8)
    parser.add_argument("--batch-size-p", type=int, default=16)
    parser.add_argument("--lr-p", type=float, default=1e-3)
    parser.add_argument("--epochs-g", type=int, default=80)
    parser.add_argument("--batch-size-g", type=int, default=64)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--l-pred", type=float, default=1.0)
    parser.add_argument("--l-simi", type=float, default=1.0)
    parser.add_argument("--l-simi-thresh", type=float, default=0.4)
    parser.add_argument("--l-potential", type=float, default=1.0)
    parser.add_argument("--l-potential-thresh", type=float, default=0.4)
    parser.add_argument("--preview-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = _build_calculator(args)
    evaluator = _train_evaluator(args, calculator)

    generator = AlphaForgeGeneratorLSTM(
        latent_size=args.latent_size,
        d_model=args.hidden_g,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(args.device)
    generator.initialize_parameters()
    tokenizer = AlphaForgeTokenizer(max_len=args.max_len)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr_g)

    best = {
        "pred_mean": float("-inf"),
        "epoch": -1,
        "preview": None,
    }

    print("device =", args.device)
    print("metric =", args.metric)
    print("generator_training = start")

    for epoch in range(args.epochs_g):
        generator.train()
        z1 = torch.randn(args.batch_size_g, args.latent_size, device=args.device)
        z2 = torch.randn(args.batch_size_g, args.latent_size, device=args.device)

        out1 = generator.forward_masked_logits(z1)
        out2 = generator.forward_masked_logits(z2)

        onehot_1 = F.gumbel_softmax(out1.masked_logits, hard=True, dim=-1)
        onehot_2 = F.gumbel_softmax(out2.masked_logits, hard=True, dim=-1)

        pred_1, latent_1 = evaluator.model(onehot_1, return_latent=True)
        _, latent_2 = evaluator.model(onehot_2, return_latent=True)

        loss_pred = -pred_1.mean()
        loss_simi = _loss_simi(onehot_1, onehot_2, args.l_simi_thresh)
        loss_potential = _loss_potential(latent_1, latent_2, args.l_potential_thresh)

        loss = (
            args.l_pred * loss_pred
            + args.l_simi * loss_simi
            + args.l_potential * loss_potential
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs_g - 1:
            generator.eval()
            with torch.no_grad():
                preview_z = torch.randn(args.preview_size, args.latent_size, device=args.device)
                preview = generator.forward_masked_logits(preview_z)
                preview_onehot = F.gumbel_softmax(preview.masked_logits, hard=True, dim=-1)
                preview_pred = evaluator.model(preview_onehot).squeeze(-1)
                preview_ids = preview_onehot.argmax(dim=-1)
                pred_mean = float(preview_pred.mean().item())
                decoded = _decode_samples(tokenizer, preview_ids, calculator, args.metric, limit=args.preview_size)
                valid = [item for item in decoded if item.valid]
                valid_rate = len(valid) / max(len(decoded), 1)
                mean_true_score = (
                    sum(item.score for item in valid if item.score is not None) / max(len(valid), 1)
                    if valid else 0.0
                )
                best_true = max((item.score for item in valid if item.score is not None), default=float("-inf"))

            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"pred_mean={pred_mean:.4f} "
                f"valid_rate={valid_rate:.3f} "
                f"mean_true_{args.metric}={mean_true_score:.4f} "
                f"best_true_{args.metric}={best_true:.4f}"
            )

            if pred_mean > best["pred_mean"]:
                best["pred_mean"] = pred_mean
                best["epoch"] = epoch
                best["preview"] = decoded

    print("--- best preview by predictor score ---")
    print("best_epoch =", best["epoch"])
    print("best_pred_mean =", best["pred_mean"])
    if best["preview"] is not None:
        for idx, item in enumerate(best["preview"]):
            if item.valid:
                print(f"[{idx}] valid | score={item.score:.6f} | expr={item.expr_str}")
            else:
                print(f"[{idx}] invalid")


if __name__ == "__main__":
    main()
