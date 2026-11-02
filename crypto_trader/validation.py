"""Validação out-of-sample com walk-forward.

Processo para cada janela de teste k:
  1. Otimiza os parâmetros SOMENTE com os dados anteriores à janela (treino expansível).
  2. Aplica os parâmetros escolhidos à janela seguinte, nunca vista no treino.
  3. Compara com buy & hold no mesmo período.

Uma estratégia só é APROVADA se o desempenho agregado fora da amostra cumprir todos os
critérios de `Criteria`. Os números in-sample não contam para a aprovação.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .backtest import BacktestResult, Costs, buy_and_hold, run_backtest, sharpe_ratio
from .data import Candle
from .strategies import STRATEGIES


@dataclass(frozen=True)
class Criteria:
    min_oos_return: float = 0.0  # retorno composto fora da amostra, já líquido de custos
    min_oos_sharpe: float = 0.5  # Sharpe anualizado fora da amostra
    max_oos_drawdown: float = 0.35  # pior drawdown em qualquer janela de teste
    min_oos_trades: int = 10  # número mínimo de operações fora da amostra
    min_profitable_folds: float = 0.6  # fração das janelas de teste com retorno positivo
    min_train_trades: int = 5  # número mínimo de operações para um parâmetro ser elegível no treino


@dataclass
class FoldResult:
    start: int
    end: int
    params: dict | None
    result: BacktestResult | None
    buy_hold: BacktestResult


@dataclass
class Report:
    strategy: str
    folds: list[FoldResult] = field(default_factory=list)
    oos_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_trades: int = 0
    profitable_folds: float = 0.0
    bh_oos_return: float = 0.0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)


def _evaluate(strategy, candles, params, costs, initial_cash, ppy, start=0, end=None):
    """Roda a estratégia usando apenas candles[:end] e backtesta candles[start:end]."""
    end = len(candles) if end is None else end
    signals = strategy.signals(candles[:end], params)
    targets = [0] + signals[: end - 1]  # sinal do candle i executa na abertura de i+1
    return run_backtest(candles[start:end], targets[start:end], costs, initial_cash, ppy)


def optimize(strategy, candles, end, costs, initial_cash, ppy, min_trades):
    """Escolhe os parâmetros de maior Sharpe usando somente candles[:end]."""
    best_params, best_score = None, -math.inf
    for params in strategy.param_sets():
        res = _evaluate(strategy, candles, params, costs, initial_cash, ppy, 0, end)
        if res.metrics["trades"] < min_trades:
            continue
        score = res.metrics["sharpe"]
        if score > best_score:
            best_params, best_score = params, score
    return best_params


def walk_forward(
    strategy_name: str,
    candles: list[Candle],
    folds: int = 5,
    costs: Costs = Costs(),
    initial_cash: float = 10_000.0,
    periods_per_year: float = 365.0,
    min_train_trades: int = 5,
) -> list[FoldResult]:
    if folds < 2:
        raise ValueError("folds deve ser >= 2")
    strategy = STRATEGIES[strategy_name]
    n = len(candles)
    bounds = [n * k // folds for k in range(folds + 1)]
    results: list[FoldResult] = []
    for k in range(1, folds):
        start, end = bounds[k], bounds[k + 1]
        bh = buy_and_hold(candles[start:end], costs, initial_cash, periods_per_year)
        params = optimize(strategy, candles, start, costs, initial_cash, periods_per_year, min_train_trades)
        if params is None:
            results.append(FoldResult(start, end, None, None, bh))
            continue
        res = _evaluate(strategy, candles, params, costs, initial_cash, periods_per_year, start, end)
        results.append(FoldResult(start, end, params, res, bh))
    return results


def validate(
    strategy_name: str,
    candles: list[Candle],
    folds: int = 5,
    criteria: Criteria = Criteria(),
    costs: Costs = Costs(),
    initial_cash: float = 10_000.0,
    periods_per_year: float = 365.0,
) -> Report:
    folds_res = walk_forward(
        strategy_name, candles, folds, costs, initial_cash, periods_per_year,
        criteria.min_train_trades,
    )
    report = Report(strategy=strategy_name, folds=folds_res)
    done = [f for f in folds_res if f.result is not None]
    if not done:
        report.reasons.append("nenhum parâmetro elegível no treino (poucas operações)")
        return report

    oos_returns = [r for f in done for r in f.result.bar_returns]
    report.oos_return = math.prod(1 + f.result.metrics["total_return"] for f in done) - 1
    report.oos_sharpe = sharpe_ratio(oos_returns, periods_per_year)
    report.oos_max_drawdown = max(abs(f.result.metrics["max_drawdown"]) for f in done)
    report.oos_trades = sum(int(f.result.metrics["trades"]) for f in done)
    report.profitable_folds = sum(f.result.metrics["total_return"] > 0 for f in done) / len(done)
    report.bh_oos_return = math.prod(1 + f.buy_hold.metrics["total_return"] for f in done) - 1

    if report.oos_return <= criteria.min_oos_return:
        report.reasons.append(f"retorno fora da amostra {report.oos_return:.2%} <= {criteria.min_oos_return:.2%}")
    if report.oos_sharpe < criteria.min_oos_sharpe:
        report.reasons.append(f"Sharpe fora da amostra {report.oos_sharpe:.2f} < {criteria.min_oos_sharpe}")
    if report.oos_max_drawdown > criteria.max_oos_drawdown:
        report.reasons.append(
            f"drawdown máximo {report.oos_max_drawdown:.2%} > {criteria.max_oos_drawdown:.2%}"
        )
    if report.oos_trades < criteria.min_oos_trades:
        report.reasons.append(f"operações fora da amostra {report.oos_trades} < {criteria.min_oos_trades}")
    if report.profitable_folds < criteria.min_profitable_folds:
        report.reasons.append(
            f"janelas lucrativas {report.profitable_folds:.0%} < {criteria.min_profitable_folds:.0%}"
        )

    report.passed = not report.reasons
    return report
