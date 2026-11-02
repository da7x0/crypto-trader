"""Estratégias de trading long-only (spot).

Cada estratégia recebe os candles e devolve uma lista de sinais 0/1 do mesmo tamanho:
sinal[i] = posição desejada AO FECHAMENTO do candle i. A execução acontece na abertura
do candle i+1 (ver validation/backtest), então o sinal nunca usa informação futura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

from .data import Candle
from .indicators import prior_max, prior_min, rsi, sma


def sma_cross(candles: list[Candle], fast: int, slow: int) -> list[int]:
    """Comprado enquanto a média móvel rápida estiver acima da lenta."""
    close = [c.close for c in candles]
    f = sma(close, fast)
    s = sma(close, slow)
    return [
        1 if (fi is not None and si is not None and fi > si) else 0
        for fi, si in zip(f, s)
    ]


def donchian_breakout(candles: list[Candle], entry: int, exit: int) -> list[int]:
    """Estilo Turtle: compra quando o fechamento rompe a máxima das últimas `entry` barras;
    vende quando rompe a mínima das últimas `exit` barras."""
    close = [c.close for c in candles]
    up = prior_max([c.high for c in candles], entry)
    down = prior_min([c.low for c in candles], exit)
    held = 0
    out: list[int] = []
    for i in range(len(candles)):
        if not held:
            if up[i] is not None and close[i] > up[i]:
                held = 1
        elif down[i] is not None and close[i] < down[i]:
            held = 0
        out.append(held)
    return out


def rsi_reversion(
    candles: list[Candle], period: int, buy_below: float, sell_above: float
) -> list[int]:
    """Reversão à média: compra quando o RSI cai abaixo de `buy_below`; vende ao subir acima de `sell_above`."""
    r = rsi([c.close for c in candles], period)
    held = 0
    out: list[int] = []
    for x in r:
        if x is not None:
            if not held and x < buy_below:
                held = 1
            elif held and x > sell_above:
                held = 0
        out.append(held)
    return out


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    fn: Callable[..., list[int]]
    grid: dict[str, list] = field(default_factory=dict)
    valid: Callable[[dict], bool] = lambda params: True

    def param_sets(self):
        keys = list(self.grid)
        for values in product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            if self.valid(params):
                yield params

    def signals(self, candles: list[Candle], params: dict) -> list[int]:
        return self.fn(candles, **params)


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in [
        Strategy(
            name="sma_cross",
            description="Cruzamento de médias móveis (SMA rápida acima da lenta)",
            fn=sma_cross,
            grid={"fast": [5, 10, 20, 30, 50], "slow": [50, 100, 150, 200]},
            valid=lambda p: p["fast"] < p["slow"],
        ),
        Strategy(
            name="donchian",
            description="Rompimento de canal de Donchian (estilo Turtle)",
            fn=donchian_breakout,
            grid={"entry": [20, 40, 55, 100], "exit": [10, 20, 30]},
            valid=lambda p: p["exit"] < p["entry"],
        ),
        Strategy(
            name="rsi_reversion",
            description="Reversão à média com RSI",
            fn=rsi_reversion,
            grid={"period": [7, 14], "buy_below": [20, 30], "sell_above": [55, 70]},
        ),
    ]
}
