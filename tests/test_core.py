import tempfile
import unittest
from pathlib import Path

from crypto_trader.backtest import Costs, buy_and_hold, run_backtest
from crypto_trader.cli import main
from crypto_trader.data import Candle, load_csv, save_csv, synthetic_candles
from crypto_trader.indicators import prior_max, prior_min, rsi, sma
from crypto_trader.strategies import STRATEGIES
from crypto_trader.validation import _evaluate, optimize, validate, walk_forward

NO_COST = Costs(fee=0.0, slippage=0.0)


def flat_candles(prices, opens=None):
    opens = opens or prices
    return [Candle(i, o, max(o, p), min(o, p), p, 1.0) for i, (o, p) in enumerate(zip(opens, prices))]


class Indicators(unittest.TestCase):
    def test_sma(self):
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, [None, None, 2.0, 3.0, 4.0])

    def test_rsi_extremes(self):
        up = rsi([float(i) for i in range(1, 30)], 14)
        self.assertEqual(up[-1], 100.0)
        flat = rsi([5.0] * 30, 14)
        self.assertEqual(flat[-1], 50.0)

    def test_prior_windows_exclude_current_bar(self):
        vals = [1, 5, 2, 9, 3]
        self.assertEqual(prior_max(vals, 2), [None, None, 5, 5, 9])
        self.assertEqual(prior_min(vals, 2), [None, None, 1, 2, 2])


class BacktestMechanics(unittest.TestCase):
    def test_fills_at_next_open_with_slippage(self):
        candles = flat_candles([10, 11, 12, 13], opens=[10, 10, 12, 14])
        costs = Costs(fee=0.0, slippage=0.01)
        # sinal 1 executado na abertura do candle 1, saída na abertura do candle 3
        res = run_backtest(candles, [0, 1, 1, 0], costs, initial_cash=1000.0)
        self.assertEqual(len(res.trades), 1)
        t = res.trades[0]
        self.assertAlmostEqual(t.entry_price, 10 * 1.01)
        self.assertAlmostEqual(t.exit_price, 14 * 0.99)
        self.assertEqual((t.entry_index, t.exit_index), (1, 3))

    def test_buy_and_hold_no_costs_equals_price_ratio(self):
        candles = synthetic_candles(200, seed=3)
        res = buy_and_hold(candles, NO_COST, initial_cash=1000.0)
        expected = 1000.0 * candles[-1].close / candles[0].open
        self.assertAlmostEqual(res.equity[-1], expected, places=6)

    def test_max_drawdown(self):
        prices = [100, 200, 100, 150]
        res = run_backtest(flat_candles(prices), [1, 1, 1, 1], NO_COST, initial_cash=100.0)
        self.assertAlmostEqual(res.metrics["max_drawdown"], -0.5)

    def test_fees_reduce_return(self):
        candles = synthetic_candles(100, seed=5)
        targets = [1] * len(candles)
        free = run_backtest(candles, targets, NO_COST).metrics["total_return"]
        paid = run_backtest(candles, targets, Costs(0.001, 0.0005)).metrics["total_return"]
        self.assertLess(paid, free)


class NoLookahead(unittest.TestCase):
    def test_signals_are_causal(self):
        candles = synthetic_candles(400, seed=11)
        for strat in STRATEGIES.values():
            params = next(strat.param_sets())
            full = strat.signals(candles, params)
            for k in (50, 137, 300):
                prefix = strat.signals(candles[:k], params)
                self.assertEqual(prefix, full[:k], f"{strat.name} depende de dados futuros")

    def test_fold_result_ignores_data_after_window(self):
        candles = synthetic_candles(600, seed=2)
        strat = STRATEGIES["donchian"]
        params = {"entry": 20, "exit": 10}
        start, end = 300, 450
        base = _evaluate(strat, candles, params, NO_COST, 1000.0, 365.0, start, end)
        # altera apenas o futuro além da janela de teste
        tampered = candles[:end] + [
            Candle(c.ts, c.open * 10, c.high * 10, c.low * 10, c.close * 10, c.volume)
            for c in candles[end:]
        ]
        again = _evaluate(strat, tampered, params, NO_COST, 1000.0, 365.0, start, end)
        self.assertEqual(base.equity, again.equity)

    def test_optimize_ignores_data_after_training_end(self):
        candles = synthetic_candles(600, seed=4)
        strat = STRATEGIES["sma_cross"]
        cut = 300
        tampered = candles[:cut] + [
            Candle(c.ts, c.open * 5, c.high * 5, c.low * 5, c.close * 5, c.volume)
            for c in candles[cut:]
        ]
        a = optimize(strat, candles, cut, NO_COST, 1000.0, 365.0, 1)
        b = optimize(strat, tampered, cut, NO_COST, 1000.0, 365.0, 1)
        self.assertEqual(a, b)


class Validation(unittest.TestCase):
    def test_walk_forward_produces_out_of_sample_folds(self):
        candles = synthetic_candles(800, seed=7)
        folds = walk_forward("sma_cross", candles, folds=4)
        self.assertEqual(len(folds), 3)
        for f in folds:
            self.assertGreater(f.start, 0)
            self.assertLessEqual(f.end, len(candles))

    def test_validate_returns_consistent_report(self):
        candles = synthetic_candles(800, seed=9)
        rep = validate("rsi_reversion", candles, folds=4)
        self.assertEqual(rep.passed, not rep.reasons)
        self.assertGreaterEqual(rep.profitable_folds, 0.0)
        self.assertLessEqual(rep.profitable_folds, 1.0)


class DataIO(unittest.TestCase):
    def test_csv_roundtrip(self):
        candles = synthetic_candles(20, seed=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            save_csv(candles, path)
            loaded = load_csv(path)
        self.assertEqual(len(loaded), len(candles))
        self.assertAlmostEqual(loaded[5].close, candles[5].close)

    def test_rejects_inconsistent_candle(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            path.write_text("timestamp,open,high,low,close,volume\n1,10,9,8,10,1\n")
            with self.assertRaises(ValueError):
                load_csv(path)


class Cli(unittest.TestCase):
    def test_synth_validate_and_signal(self):
        with tempfile.TemporaryDirectory() as d:
            data = str(Path(d) / "s.csv")
            self.assertEqual(main(["synth", "--n", "600", "--out", data]), 0)
            # exit 2 = nenhuma estratégia aprovada (esperado em passeio aleatório); ambos são válidos
            self.assertIn(main(["validate", "--data", data, "--folds", "4"]), (0, 2))
            self.assertEqual(
                main(["signal", "--data", data, "--strategy", "donchian", "--params", "entry=20,exit=10"]), 0
            )
            self.assertEqual(
                main(["backtest", "--data", data, "--strategy", "sma_cross", "--params", "fast=10,slow=50"]), 0
            )


if __name__ == "__main__":
    unittest.main()
