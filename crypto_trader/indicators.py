"""Indicadores técnicos. Todos retornam listas do mesmo tamanho da entrada.

Posições sem histórico suficiente valem None. Cada valor na posição i usa apenas dados até i
(inclusive), por isso nenhum indicador introduz lookahead.
"""

from __future__ import annotations


def sma(values: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if n < 1:
        raise ValueError("n deve ser >= 1")
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def rsi(values: list[float], n: int) -> list[float | None]:
    """RSI com suavização de Wilder."""
    out: list[float | None] = [None] * len(values)
    if n < 1 or len(values) <= n:
        return out
    gain = loss = 0.0
    for i in range(1, n + 1):
        d = values[i] - values[i - 1]
        gain += max(d, 0.0)
        loss += max(-d, 0.0)
    gain /= n
    loss /= n
    out[n] = _rsi_value(gain, loss)
    for i in range(n + 1, len(values)):
        d = values[i] - values[i - 1]
        gain = (gain * (n - 1) + max(d, 0.0)) / n
        loss = (loss * (n - 1) + max(-d, 0.0)) / n
        out[i] = _rsi_value(gain, loss)
    return out


def _rsi_value(gain: float, loss: float) -> float:
    if loss == 0.0:
        return 100.0 if gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def prior_max(values: list[float], n: int) -> list[float | None]:
    """Máximo das n barras ANTERIORES a i (exclui a barra i)."""
    out: list[float | None] = [None] * len(values)
    for i in range(n, len(values)):
        out[i] = max(values[i - n : i])
    return out


def prior_min(values: list[float], n: int) -> list[float | None]:
    """Mínimo das n barras ANTERIORES a i (exclui a barra i)."""
    out: list[float | None] = [None] * len(values)
    for i in range(n, len(values)):
        out[i] = min(values[i - n : i])
    return out
