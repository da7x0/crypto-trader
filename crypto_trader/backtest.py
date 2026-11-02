"""Motor de backtest long-only com taxas e slippage.

Regra de execução: `targets[i]` é a posição desejada que é executada na ABERTURA do candle i.
Quem monta `targets` deve ter deslocado os sinais de 1 barra (ver validation._evaluate),
de modo que o sinal do fechamento de i-1 seja executado na abertura de i.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .data import Candle


@dataclass(frozen=True)
class Costs:
    fee: float = 0.001  # taxa por lado (0,10%, faixa típica de taker em corretoras spot)
    slippage: float = 0.0005  # 0,05% contra você em cada execução


@dataclass(frozen=True)
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    pnl: float  # lucro/prejuízo em moeda de cotação, já descontadas as taxas
    return_pct: float


@dataclass
class BacktestResult:
    equity: list[float]  # índice 0 = capital inicial; índice i+1 = fechamento do candle i
    bar_returns: list[float]
    trades: list[Trade]
    metrics: dict[str, float]


def run_backtest(
    candles: list[Candle],
    targets: list[int],
    costs: Costs = Costs(),
    initial_cash: float = 10_000.0,
    periods_per_year: float = 365.0,
) -> BacktestResult:
    if len(candles) != len(targets):
        raise ValueError("candles e targets devem ter o mesmo tamanho")
    if not candles:
        raise ValueError("Sem dados para o backtest")

    cash = float(initial_cash)
    qty = 0.0
    entry_value = 0.0
    entry_price = 0.0
    entry_idx = 0
    equity = [cash]
    trades: list[Trade] = []
    exposed_bars = 0

    for i, (candle, target) in enumerate(zip(candles, targets)):
        if target and qty == 0.0 and cash > 0.0:
            price = candle.open * (1 + costs.slippage)
            entry_value = cash
            qty = cash / (price * (1 + costs.fee))
            cash = 0.0
            entry_price, entry_idx = price, i
        elif not target and qty > 0.0:
            price = candle.open * (1 - costs.slippage)
            cash, trade = _close(qty, price, costs, entry_value, entry_price, entry_idx, i)
            trades.append(trade)
            qty = 0.0

        if qty > 0.0:
            exposed_bars += 1
        equity.append(cash + qty * candle.close)

    if qty > 0.0:  # fecha posição aberta no último fechamento, para contabilizar as taxas
        last = len(candles) - 1
        price = candles[-1].close * (1 - costs.slippage)
        cash, trade = _close(qty, price, costs, entry_value, entry_price, entry_idx, last)
        trades.append(trade)
        qty = 0.0
        equity[-1] = cash

    bar_returns = [
        equity[i + 1] / equity[i] - 1.0 if equity[i] > 0 else 0.0
        for i in range(len(equity) - 1)
    ]
    metrics = _metrics(equity, bar_returns, trades, exposed_bars, len(candles), periods_per_year)
    return BacktestResult(equity, bar_returns, trades, metrics)


def buy_and_hold(
    candles: list[Candle],
    costs: Costs = Costs(),
    initial_cash: float = 10_000.0,
    periods_per_year: float = 365.0,
) -> BacktestResult:
    """Referência: compra na primeira abertura e segura até o fim."""
    return run_backtest(candles, [1] * len(candles), costs, initial_cash, periods_per_year)


def sharpe_ratio(returns: list[float], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    sd = statistics.stdev(returns)
    if sd == 0.0:
        return 0.0
    return statistics.fmean(returns) / sd * math.sqrt(periods_per_year)


def _close(qty, price, costs, entry_value, entry_price, entry_idx, exit_idx):
    proceeds = qty * price * (1 - costs.fee)
    trade = Trade(
        entry_index=entry_idx,
        exit_index=exit_idx,
        entry_price=entry_price,
        exit_price=price,
        pnl=proceeds - entry_value,
        return_pct=proceeds / entry_value - 1.0,
    )
    return proceeds, trade


def _metrics(equity, bar_returns, trades, exposed_bars, n_bars, periods_per_year):
    start, end = equity[0], equity[-1]
    total_return = end / start - 1.0
    years = n_bars / periods_per_year
    cagr = (end / start) ** (1 / years) - 1.0 if years > 0 and end > 0 else -1.0

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, v / peak - 1.0)

    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl <= 0)
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = math.inf if gross_win > 0 else 0.0
    wins = sum(1 for t in trades if t.pnl > 0)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe_ratio(bar_returns, periods_per_year),
        "max_drawdown": max_dd,
        "trades": float(len(trades)),
        "win_rate": wins / len(trades) if trades else 0.0,
        "profit_factor": profit_factor,
        "exposure": exposed_bars / n_bars,
    }
