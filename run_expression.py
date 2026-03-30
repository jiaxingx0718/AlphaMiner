from __future__ import annotations

from code import interact
from pathlib import Path

from data.datatotensor import DataToTensorConfig, StockData
from expressions.expression import (
    ABS,
    ADD,
    AMOUNT,
    BINMAX,
    BINMIN,
    CLOSE,
    Constant,
    CSRANK,
    DELTA,
    DIV,
    EMA,
    EXP,
    HIGH,
    KURT,
    LOG,
    LOW,
    MA,
    MAX,
    MED,
    MIN,
    MUL,
    OPEN,
    PAST,
    POW,
    RANK,
    SIGN,
    SKEW,
    STD,
    SUB,
    SUM,
    VOLUME,
    WMA,
)


def build_data() -> StockData:
    project_root = Path(__file__).resolve().parent
    config = DataToTensorConfig(
        data_dir=project_root / "data" / "cleaned" / "daily_cleaned",
        device="cuda:0",
    )
    return StockData(
        config=config,
        selected_stock_ids=["000001", "000002", "000004", "000006", "000007"],
    )


if __name__ == "__main__":
    data = build_data()

    banner = """
AlphaMiner expression playground

Preloaded:
- data
- basic features: OPEN CLOSE HIGH LOW VOLUME AMOUNT
- unary/binary/rolling/past operators

Examples:
expr = ADD(LOG(CLOSE), 5)
value = expr.evaluate(data)
print(value.shape)

expr = MA(ADD(CLOSE, OPEN), 5)
value = expr.evaluate(data)
print(value[:8, :2])
""".strip()

    interact(
        banner=banner,
        local=locals(),
    )
