# crypto-trader

Biblioteca em Python puro (sem dependências externas) para **backtest e validação out-of-sample** de estratégias de criptomoedas, com um CLI que mostra o sinal atual.

> Este projeto **não envia ordens** para corretoras. Ele serve para pesquisar, testar e validar estratégias. Backtests e aprovações fora da amostra reduzem o risco de sobreajuste, mas não garantem lucro. Teste em modo simulado antes de qualquer capital real.

## Estratégias

| Nome | Lógica | Parâmetros da grade |
|---|---|---|
| `sma_cross` | Comprado enquanto a média rápida está acima da lenta | `fast`, `slow` |
| `donchian` | Estilo Turtle: compra no rompimento da máxima de `entry` barras, vende no rompimento da mínima de `exit` barras | `entry`, `exit` |
| `rsi_reversion` | Compra quando o RSI cai abaixo de `buy_below`, vende quando sobe acima de `sell_above` | `period`, `buy_below`, `sell_above` |

## Requisitos

Python 3.10 ou superior. Nenhuma biblioteca externa é necessária.

## Uso

Todos os comandos são executados a partir da raiz do repositório.

```bash
# 1. Baixar candles diários da Binance (últimos ~4 anos)
python -m crypto_trader fetch --symbol BTCUSDT --interval 1d --days 1500 --out data/BTCUSDT_1d.csv

# 2. Backtest com parâmetros fixos
python -m crypto_trader backtest --data data/BTCUSDT_1d.csv --strategy donchian --params entry=55,exit=20

# 3. Validação walk-forward de todas as estratégias, com veredito PASS/FAIL
python -m crypto_trader validate --data data/BTCUSDT_1d.csv data/ETHUSDT_1d.csv --strategy all

# 4. Sinal no último candle fechado (não envia ordens)
python -m crypto_trader signal --data data/BTCUSDT_1d.csv --strategy sma_cross --params fast=20,slow=100
```

Os comandos de backtest aceitam `--fee` (padrão 0,10% por lado), `--slippage` (padrão 0,05% por execução), `--capital` e `--interval` (`1h`, `4h` ou `1d`).

O comando `validate` retorna código `0` se alguma estratégia for aprovada e `2` se nenhuma for.

Formato do CSV: `timestamp,open,high,low,close,volume`, com `timestamp` em milissegundos.

Para testar o pipeline sem rede:

```bash
python -m crypto_trader synth --out data/synth.csv
python -m crypto_trader validate --data data/synth.csv
```

Dados sintéticos não têm padrões reais de mercado. Qualquer resultado com eles serve só para verificar o funcionamento do código.

## Metodologia

**Sem lookahead.** O sinal do fechamento do candle `i` é executado na abertura do candle `i+1`. Os indicadores usam apenas dados até `i`. Há testes que verificam isso: alterar o futuro não muda o resultado.

**Custos realistas.** Cada compra e venda paga taxa e slippage. Posições abertas no fim do período são liquidadas no último fechamento, com custos.

**Walk-forward.** O histórico é dividido em `folds` blocos. Para cada bloco de teste, os parâmetros são escolhidos apenas com os dados anteriores (treino expansível) e aplicados ao bloco seguinte, nunca visto na otimização. A escolha de parâmetros usa o Sharpe no treino.

**Critérios de aprovação** (todos precisam ser atendidos, sobre os blocos de teste agregados):

| Critério | Padrão | Flag |
|---|---|---|
| Retorno composto fora da amostra, líquido de custos | > 0% | fixo |
| Sharpe anualizado fora da amostra | ≥ 0,5 | `--min-sharpe` |
| Pior drawdown em um bloco de teste | ≤ 35% | `--max-drawdown` |
| Operações fora da amostra | ≥ 10 | `--min-trades` |
| Blocos de teste com retorno positivo | ≥ 60% | fixo |

O buy & hold é mostrado como referência, mas não é critério de aprovação.

## Limitações conhecidas

- **Sobreajuste por seleção.** Testar muitas combinações de parâmetros e estratégias e escolher a que "passou" infla os resultados. A validação fora da amostra mitiga isso, mas só é honesta se os critérios não forem ajustados depois de ver os resultados.
- **Falsos positivos.** Em passeios aleatórios sem drift, os critérios padrão aprovam cerca de 1 em cada 10 séries (medido com 40 séries sintéticas por estratégia). Trate uma aprovação como sinal para investigar, não como prova.
- **Custos simplificados.** Taxa fixa e slippage fixo. Não há modelo de liquidez do livro de ofertas nem de falhas de execução.
- **Apenas long e spot.** Sem alavancagem, venda a descoberto ou derivativos.
- **Sem execução ao vivo.** Enviar ordens exige chaves de API, gestão de risco e testes em modo simulado. Não está implementado de propósito.

## Testes

```bash
python -m unittest -v
```

Os testes cobrem indicadores, mecânica de execução, métricas, causalidade dos sinais e de cada janela de validação, e o CLI.
