from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calculator.calculator import StockDataCalculator
from data.datatotensor import DataToTensorConfig, StockData
from expressions.expression import CLOSE, FORWARDRET
from RL.evalcore import AlphaEnvCore
from RL.policy import LSTMSharedNet
from RL.wrapper import AlphaEnvWrapper


@dataclass
class TrainConfig:
    """
    训练超参数
    """

    # 数据层和目标函数
    data_dir: str
    stock_ids: Optional[Sequence[str]] = None
    eval_metric: str = "ic"
    forward_window: int = 5
    max_invalid: Optional[int] = 1_000_000
    cs_max_invalid: Optional[int] = 1_000_000
    winsorize: bool = True
    normalize: bool = True

    # 分数层
    max_expr_length: int = 15
    step_reward: float = 0.0
    invalid_reward: float = -1.0
    invalid_eval_reward: float = 0.0
    print_expr: bool = False

    # 编码器结构
    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.1

    # 强化学习
    total_timesteps: int = 200_000
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5

    # 运行和日志
    device: str = "cuda:0"
    seed: int = 0
    log_dir: str = "./out/tensorboard"
    save_dir: str = "./out/checkpoints"
    tb_log_name: str = "alphaminer_lstm"
    checkpoint_freq: int = 25_000


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _mask_fn(env: AlphaEnvWrapper):
    """
    sb3-contrib 的 ActionMasker 会在每步采样前调用这个函数。
    """
    return env.action_mask()


def build_calculator(config: TrainConfig) -> StockDataCalculator:
    """
    构造单因子评估器

    这一层负责把:
    - 原始股票面板数据
    - target 表达式
    - winsorize / normalize 等预处理

    统一包装成 reward 评估入口，供 AlphaEnvCore 调用
    """
    data_cfg = DataToTensorConfig(
        data_dir=Path(config.data_dir),
        device=_resolve_device(config.device),
    )
    data = StockData(config=data_cfg, selected_stock_ids=config.stock_ids)

    target = FORWARDRET(CLOSE, config.forward_window)
    return StockDataCalculator(
        data=data,
        target=target,
        max_invalid=config.max_invalid,
        cs_max_invalid=config.cs_max_invalid,
        winsorize=config.winsorize,
        normalize=config.normalize,
    )


def build_env(cfg: TrainConfig) -> AlphaEnvWrapper:
    """
    构造 RL 环境并接上 action mask

    分工如下:
    - AlphaEnvCore: 表达式构树、合法性检查、reward
    - AlphaEnvWrapper: action/int 与 token 的映射，维护 observation
    - ActionMasker: 在采样前屏蔽当前状态下不合法的动作
    """
    calculator = build_calculator(cfg)
    core = AlphaEnvCore(
        calculator=calculator,
        max_expr_length=cfg.max_expr_length,
        step_reward=cfg.step_reward,
        invalid_reward=cfg.invalid_reward,
        invalid_eval_reward=cfg.invalid_eval_reward,
        print_expr=cfg.print_expr,
        eval_metric=cfg.eval_metric,
    )
    env = AlphaEnvWrapper(core)
    return ActionMasker(env, _mask_fn)


def build_model(cfg: TrainConfig, env: ActionMasker) -> MaskablePPO:
    """
    这里基本照 AlphaGen 的 rl.py 写法:
    - 使用 MaskablePPO
    - policy 仍用 "MlpPolicy"
    - 只替换 features_extractor_class 为 LSTMSharedNet
    - actor/critic 的默认 MLP 仍交给 SB3 自动构造

    也就是说，本项目里自定义的是:
        s_t -> h_t
    而不是完整重写 actor / critic。

    后续结构仍由 SB3 自动补齐:
        h_t -> policy logits
        h_t -> value
    """
    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            device=torch.device(_resolve_device(cfg.device)),
        ),
    )

    return MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        clip_range=cfg.clip_range,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
        device=_resolve_device(cfg.device),
        verbose=1,
    )


def save_run_config(cfg: TrainConfig, run_dir: Path) -> None:
    """
    保存当前训练配置，便于之后回溯 checkpoint 对应的实验参数。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def main(cfg: TrainConfig) -> None:
    """
    训练入口。

    完整流程:
    1. 生成本次 run 目录
    2. 保存训练配置
    3. 构造 env
    4. 构造 MaskablePPO
    5. 训练并定期保存 checkpoint
    6. 保存最终模型
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.tb_log_name}_{timestamp}"
    run_dir = Path(cfg.save_dir) / run_name
    save_run_config(cfg, run_dir)

    env = build_env(cfg)
    model = build_model(cfg, env)

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=str(run_dir),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))

    print("Training finished.")
    print(f"Saved to: {run_dir}")
    print("TensorBoard:")
    print(f"  tensorboard --logdir {cfg.log_dir}")


def parse_args() -> TrainConfig:
    """
    将命令行参数映射到 TrainConfig。

    这里保留 AlphaGen 中最常用的一组入口，同时把环境、模型、PPO 参数都显式暴露。
    """
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data" / "cleaned" / "daily_cleaned"

    parser = argparse.ArgumentParser(description="Train AlphaMiner with MaskablePPO + LSTMSharedNet")

    # 数据与 target
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--stock-ids", nargs="*", default=None)
    parser.add_argument("--eval-metric", type=str, default="ic", choices=["ic", "rankic"])
    parser.add_argument("--forward-window", type=int, default=5)
    parser.add_argument("--max-invalid", type=int, default=1_000_000)
    parser.add_argument("--cs-max-invalid", type=int, default=1_000_000)
    parser.add_argument("--no-winsorize", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")

    # 环境与 reward
    parser.add_argument("--max-expr-length", type=int, default=15)
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--invalid-eval-reward", type=float, default=0.0)
    parser.add_argument("--print-expr", action="store_true")

    # 编码器结构
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # PPO 超参数
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-5)

    # 日志与设备
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./out/tensorboard")
    parser.add_argument("--save-dir", type=str, default="./out/checkpoints")
    parser.add_argument("--tb-log-name", type=str, default="alphaminer_lstm")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)

    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        stock_ids=args.stock_ids,
        eval_metric=args.eval_metric,
        forward_window=args.forward_window,
        max_invalid=args.max_invalid,
        cs_max_invalid=args.cs_max_invalid,
        winsorize=not args.no_winsorize,
        normalize=not args.no_normalize,
        max_expr_length=args.max_expr_length,
        step_reward=args.step_reward,
        invalid_reward=args.invalid_reward,
        invalid_eval_reward=args.invalid_eval_reward,
        print_expr=args.print_expr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        learning_rate=args.learning_rate,
        adam_eps=args.adam_eps,
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        tb_log_name=args.tb_log_name,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main(parse_args())
