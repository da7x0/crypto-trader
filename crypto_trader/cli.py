"""Interface de linha de comando.

Comandos:
  fetch     baixa candles da Binance para um CSV local
  synth     gera dados sintéticos (apenas para testar o pipeline offline)
  backtest  roda uma estratégia com parâmetros fixos
  validate  walk-forward fora da amostra e veredito PASS/FAIL
  signal    mostra a posição que a estratégia indica no último candle fechado (não envia ordens)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .backtest import Costs, buy_and_hold, run_backtest
from .data import (
    INTERVAL_PERIODS_PER_YEAR,
    fetch_binance,
    load_csv,
    save_csv,
    synthetic_candles,
)
from .strategies import STRATEGIES
from .validation import Criteria, validate


def _parse_params(text: str | None) -> dict:
    if not text:
        return {}
    out: dict = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        num = float(value)
        out[key.strip()] = int(num) if num.is_integer() else num
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _ppy(interval: str) -> float:
    if interval not in INTERVAL_PERIODS_PER_YEAR:
        raise SystemExit(f"Intervalo não suportado: {interval}. Use: {', '.join(INTERVAL_PERIODS_PER_YEAR)}")
    return float(INTERVAL_PERIODS_PER_YEAR[interval])


def _costs(args) -> Costs:
    return Costs(fee=args.fee, slippage=args.slippage)


def _check_strategy(name: str) -> None:
    if name not in STRATEGIES:
        raise SystemExit(f"Estratégia desconhecida: {name}. Opções: {', '.join(STRATEGIES)}")


def cmd_fetch(args) -> int:
    try:
        candles = fetch_binance(args.symbol, args.interval, args.days, args.base_url)
    except OSError as exc:  # URLError é subclasse de OSError
        print(f"Falha ao acessar a API da Binance: {exc}. Tente --base-url ou use um CSV local.", file=sys.stderr)
        return 1
    if not candles:
        print("Nenhum candle retornado.", file=sys.stderr)
        return 1
    save_csv(candles, args.out)
    print(f"{len(candles)} candles salvos em {args.out}")
    return 0


def cmd_synth(args) -> int:
    save_csv(synthetic_candles(args.n, args.seed), args.out)
    print(f"{args.n} candles sintéticos salvos em {args.out} (NÃO são dados reais)")
    return 0


def cmd_backtest(args) -> int:
    _check_strategy(args.strategy)
    strategy = STRATEGIES[args.strategy]
    params = _required_params(args.params, strategy)
    ppy = _ppy(args.interval)
    candles = load_csv(args.data)
    costs = _costs(args)

    signals = strategy.signals(candles, params)
    targets = [0] + signals[:-1]
    res = run_backtest(candles, targets, costs, args.capital, ppy)
    bh = buy_and_hold(candles, costs, args.capital, ppy)

    print(f"Estratégia: {args.strategy} {params}")
    print(f"Período: {_fmt_ts(candles[0].ts)} até {_fmt_ts(candles[-1].ts)} ({len(candles)} candles)")
    _print_metrics(res.metrics)
    print(f"Buy & hold, retorno total: {_pct(bh.metrics['total_return'])}, Sharpe: {bh.metrics['sharpe']:.2f}")
    print("Últimas operações:")
    for t in res.trades[-5:]:
        print(
            f"  {_fmt_ts(candles[t.entry_index].ts)} -> {_fmt_ts(candles[t.exit_index].ts)} "
            f"entrada {t.entry_price:.4f} saída {t.exit_price:.4f} retorno {_pct(t.return_pct)}"
        )
    return 0


def cmd_validate(args) -> int:
    names = list(STRATEGIES) if args.strategy == ["all"] else args.strategy
    for name in names:
        _check_strategy(name)
    ppy = _ppy(args.interval)
    criteria = Criteria(
        min_oos_sharpe=args.min_sharpe,
        max_oos_drawdown=args.max_drawdown,
        min_oos_trades=args.min_trades,
    )
    costs = _costs(args)

    any_pass = False
    for path in args.data:
        candles = load_csv(path)
        print(f"\n=== {path} ({len(candles)} candles, {_fmt_ts(candles[0].ts)} a {_fmt_ts(candles[-1].ts)})")
        print(f"{'estratégia':<16}{'veredito':<9}{'ret OOS':>10}{'Sharpe':>8}{'DD máx':>9}{'trades':>8}{'janelas+':>10}{'B&H OOS':>10}")
        for name in names:
            rep = validate(name, candles, args.folds, criteria, costs, args.capital, ppy)
            any_pass |= rep.passed
            verdict = "PASS" if rep.passed else "FAIL"
            print(
                f"{name:<16}{verdict:<9}{_pct(rep.oos_return):>10}{rep.oos_sharpe:>8.2f}"
                f"{_pct(rep.oos_max_drawdown):>9}{rep.oos_trades:>8}{rep.profitable_folds:>10.0%}"
                f"{_pct(rep.bh_oos_return):>10}"
            )
            for reason in rep.reasons:
                print(f"    - reprovada: {reason}")
            for f in rep.folds:
                if f.params is not None:
                    ret = f.result.metrics["total_return"]
                    print(f"    janela [{f.start}:{f.end}] parâmetros {f.params} retorno {_pct(ret)}")

    print("\nAviso: aprovação fora da amostra reduz, mas não elimina, o risco de sobreajuste.")
    print("Dados históricos não garantem resultados futuros.")
    return 0 if any_pass else 2


def cmd_signal(args) -> int:
    _check_strategy(args.strategy)
    strategy = STRATEGIES[args.strategy]
    params = _required_params(args.params, strategy)
    candles = load_csv(args.data)
    signals = strategy.signals(candles, params)
    last = signals[-1]
    prev = signals[-2] if len(signals) > 1 else last
    c = candles[-1]
    acao = "COMPRADO (manter ou comprar na abertura do próximo candle)" if last else "FORA (manter caixa ou vender na abertura do próximo candle)"
    print(f"Estratégia: {args.strategy} {params}")
    print(f"Último candle fechado: {_fmt_ts(c.ts)} close={c.close}")
    print(f"Sinal: {acao}")
    if last != prev:
        print("Atenção: o sinal mudou neste último candle.")
    print("Nenhuma ordem foi enviada. Este comando apenas calcula o sinal.")
    return 0


def _required_params(text: str | None, strategy) -> dict:
    if not text:
        grid = ", ".join(f"{k} (ex.: {v})" for k, v in strategy.grid.items())
        raise SystemExit(f"Informe --params para {strategy.name}. Parâmetros: {grid}")
    return _parse_params(text)


def _print_metrics(m: dict) -> None:
    print(f"Retorno total: {_pct(m['total_return'])}   CAGR: {_pct(m['cagr'])}   Sharpe: {m['sharpe']:.2f}")
    print(f"Drawdown máximo: {_pct(m['max_drawdown'])}   Operações: {int(m['trades'])}   Acerto: {_pct(m['win_rate'])}")
    print(f"Fator de lucro: {m['profit_factor']:.2f}   Exposição: {_pct(m['exposure'])}")


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crypto_trader", description="Backtest e validação de estratégias cripto")
    sub = p.add_subparsers(dest="command", required=True)

    def add_cost_args(sp):
        sp.add_argument("--capital", type=float, default=10_000.0, help="capital inicial (cotação)")
        sp.add_argument("--fee", type=float, default=0.001, help="taxa por lado (0.001 = 0,1%%)")
        sp.add_argument("--slippage", type=float, default=0.0005, help="slippage por execução")
        sp.add_argument("--interval", default="1d", choices=list(INTERVAL_PERIODS_PER_YEAR))

    f = sub.add_parser("fetch", help="baixar candles da Binance")
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--interval", default="1d", choices=list(INTERVAL_PERIODS_PER_YEAR))
    f.add_argument("--days", type=int, default=1500)
    f.add_argument("--base-url", default=None, help="endpoint alternativo da Binance")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("synth", help="gerar dados sintéticos para testes")
    s.add_argument("--n", type=int, default=1500)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_synth)

    b = sub.add_parser("backtest", help="backtest com parâmetros fixos")
    b.add_argument("--data", required=True)
    b.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    b.add_argument("--params", help="ex.: fast=20,slow=100")
    add_cost_args(b)
    b.set_defaults(func=cmd_backtest)

    v = sub.add_parser("validate", help="walk-forward e veredito PASS/FAIL")
    v.add_argument("--data", nargs="+", required=True)
    v.add_argument("--strategy", nargs="+", default=["all"], help="nomes ou 'all'")
    v.add_argument("--folds", type=int, default=5)
    v.add_argument("--min-sharpe", type=float, default=Criteria.min_oos_sharpe)
    v.add_argument("--max-drawdown", type=float, default=Criteria.max_oos_drawdown)
    v.add_argument("--min-trades", type=int, default=Criteria.min_oos_trades)
    add_cost_args(v)
    v.set_defaults(func=cmd_validate)

    g = sub.add_parser("signal", help="sinal no último candle (sem ordens)")
    g.add_argument("--data", required=True)
    g.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    g.add_argument("--params")
    g.set_defaults(func=cmd_signal)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
