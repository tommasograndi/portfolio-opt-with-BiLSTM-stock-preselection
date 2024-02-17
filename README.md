# portfolio-opt-with-BiLSTM-stock-preselection

Training pipeline:
- scaler
- train-test split
- fit and predict
- compute returns
- performance = cumulative returns
- sort in descending
- choices = sorted.index[:N]
- take exc_returns.loc['2018-12-03':][choices] and compute rendimento totale (1+r_i).prod() - 1
- sum(valori ottenuti * 1/N) = rendimento portafoglio

In summary:
- scaler
- train-test split
- fit and predict
- get ranking
- calc ptfs
- plot ptf