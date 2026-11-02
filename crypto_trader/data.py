"""Carregamento, download e geração de candles OHLCV."""

from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BINANCE_BASE_URLS = ("https://api.binance.com", "https://data-api.binance.vision")

# Quantos períodos existem em um ano para cada intervalo (usado para anualizar métricas).
INTERVAL_PERIODS_PER_YEAR = {
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}

CSV_FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Candle:
    ts: int  # abertura do candle, em milissegundos desde a época Unix
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_csv(path: str | Path) -> list[Candle]:
    """Lê um CSV com colunas timestamp,open,high,low,close,volume (timestamp em ms)."""
    candles: list[Candle] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            candles.append(
                Candle(
                    ts=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    candles.sort(key=lambda c: c.ts)
    _check_candles(candles, path)
    return candles


def save_csv(candles: list[Candle], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for c in candles:
            writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])


def fetch_binance(
    symbol: str,
    interval: str = "1d",
    days: int = 1500,
    base_url: str | None = None,
    pause: float = 0.2,
) -> list[Candle]:
    """Baixa candles históricos (klines) da Binance, paginando de 1000 em 1000.

    Descarta o candle ainda em formação, para que nenhum sinal use dados incompletos.
    """
    now_ms = int(time.time() * 1000)
    cursor = now_ms - days * 86_400_000
    base = base_url or BINANCE_BASE_URLS[0]
    by_ts: dict[int, Candle] = {}

    while cursor < now_ms:
        query = urllib.parse.urlencode(
            {"symbol": symbol.upper(), "interval": interval, "startTime": cursor, "limit": 1000}
        )
        rows = _get_json(f"{base}/api/v3/klines?{query}")
        if not rows:
            break
        for k in rows:
            close_time = int(k[6])
            if close_time >= now_ms:  # candle ainda aberto
                continue
            by_ts[int(k[0])] = Candle(
                ts=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            )
        cursor = int(rows[-1][0]) + 1
        if len(rows) < 1000:
            break
        time.sleep(pause)

    return [by_ts[ts] for ts in sorted(by_ts)]


def _get_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def _check_candles(candles: list[Candle], origin: object) -> None:
    if not candles:
        raise ValueError(f"Nenhum candle em {origin}")
    for prev, cur in zip(candles, candles[1:]):
        if cur.ts == prev.ts:
            raise ValueError(f"Timestamp duplicado em {origin}: {cur.ts}")
    for c in candles:
        if not (c.low <= min(c.open, c.close) and c.high >= max(c.open, c.close)):
            raise ValueError(f"Candle inconsistente em {origin}: ts={c.ts}")
        if c.close <= 0 or c.open <= 0:
            raise ValueError(f"Preço não positivo em {origin}: ts={c.ts}")


def synthetic_candles(
    n: int = 1500,
    seed: int = 42,
    start_price: float = 100.0,
    drift: float = 0.0004,
    volatility: float = 0.03,
    step_ms: int = 86_400_000,
    start_ts: int = 1_600_000_000_000,
) -> list[Candle]:
    """Gera um passeio aleatório geométrico. Útil para testes e para ver o pipeline funcionar offline.

    Não contém padrões reais de mercado: uma estratégia aprovada aqui NÃO é evidência de nada.
    """
    rnd = random.Random(seed)
    price = start_price
    out: list[Candle] = []
    for i in range(n):
        o = price
        c = o * math.exp(drift + volatility * rnd.gauss(0.0, 1.0))
        hi = max(o, c) * (1 + abs(rnd.gauss(0.0, 0.005)))
        lo = min(o, c) * (1 - abs(rnd.gauss(0.0, 0.005)))
        out.append(Candle(start_ts + i * step_ms, o, hi, lo, c, rnd.uniform(1e3, 1e4)))
        price = c
    return out
