import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas_market_calendars as mcal
import plotly.express as px
import plotly.graph_objects as go


def price_to_returns(df: pd.DataFrame, log=False, drop_na=False) -> pd.DataFrame:
    """
    :param df: starting DataFrame with prices.
    :param log: if True, reurns log-returns. Default=True
    :param drop_na: if True, drop all rows with np.NAN values
    :return: DataFrame with returns.
    """
    if log:
        result = np.log(df / df.shift(1))
    else:
        result = (df - df.shift(1)) / df.shift(1)
    if drop_na:
        result = result.dropna()
    return result


# cumulative returns for a series 
def cumulative_returns_from_series(series: pd.Series, log=False, starting_capital=1) -> pd.Series:
    """
    :param series: pandas series of returns.
    :param log: True if returns are in logarithmic form.
    :param start: starting capital. Default=1.
    :return: pd.series of cumulative returns.
    """

    result = starting_capital * (1 + series).cumprod()

    return pd.Series(result, index = series.index)


def setup_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Used to add column date to dataframe. Sort df by date.
    :param df: starting dataframe
    :return: df with changed column
    """
    X = df.copy()
    col_name = X.columns[0]
    X.rename(columns={col_name: "Date"}, inplace=True)
    X["Date"] = pd.to_datetime(X["Date"], format="%Y%m%d")

    X.sort_values(by=["Date"], inplace=True)
    X.reset_index(drop=True, inplace=True)
    return X


def select_time_slice(df: pd.DataFrame, start: int = 20020102, end: int = 20191013) -> pd.DataFrame:
    """
    :param df: dataframe to slice
    :param start: starting day
    :param end: ending day
    :param date_column_name: name of column containing dates
    :return: slice of starting df
    """
    X = df.copy()
    start = pd.to_datetime(start, format="%Y%m%d")
    end = pd.to_datetime(end, format="%Y%m%d")
    X = X.loc[X.index <= end]
    X = X.loc[X.index >= start]
    return X


def get_full_time_stock_in_period(comp_df: pd.DataFrame) -> list:
    """
    :param comp_df: dataframe used to specify if a stock is in the market index or not
    :return: list of stock names
    """
    sum_index_series = comp_df.sum(axis=0)
    objective_value = len(comp_df)
    result = []
    for stock in sum_index_series.index:
        if sum_index_series[stock] == objective_value:
            result.append(stock)
    return result


def get_trading_dates(start_period: str = "2013-01-01", end_period: str = "2017-12-31", market: str = "EUREX") -> pd.DatetimeIndex:
    calendar = mcal.get_calendar(market)
    schedule = calendar.schedule(start_date=start_period, end_date=end_period)
    dates = mcal.date_range(schedule, frequency="1D")
    return dates.strftime('%Y-%m-%d')


def compute_market_returns(composition: pd.DataFrame, capitalization: pd.DataFrame, returns: pd.DataFrame, log=True) -> pd.Series:
    """
    Compute market return as the cap-weighted return of all the stocks.
    :param composition:
    :param capitalization:
    :param prices: price df for all the stocks composing the index []; for your period of interest

    :--RETURN: series with returns of the index (weighted by market capitalization)
    """
    # Compute weights
    weights = capitalization * composition
    weights = weights / weights.sum(axis=1).values.reshape((-1, 1))
    weights.fillna(0, inplace=True)

    weights = weights.loc[returns.index[0]:, :] #starting date of weights = starting date of returns

    weighted_returns = weights * returns
    result = pd.Series(weighted_returns.sum(axis=1), index=weights.index, name="SX5E_returns")
    return result

def get_ranking(predictions, N: list, prices : bool):
    """
    Considering the df of predictions:
    1) Calculate the cumulative returns for each stock (for the considered period)
    2) Calculate the ranking in descending order for the cumulative returns
    3) Select the top stocks among the ranking (for all top Ns)
    """

    if prices:
        returns = price_to_returns(predictions, log=True, drop_na=True)
        cum_returns = (1 + returns).prod() - 1
    
    else:
        cum_returns = (1 + predictions).prod() - 1

    ranking = cum_returns.sort_values(ascending=False).index

    portfolios = {}

    for i in N:
        portfolios[f'Top {i}'] = list(ranking[:i])

    # Return N lists with names of the top stocks according to the model's ranking
    # basically the stocks composing each portfolio with N stocks 
    return portfolios

def calc_portfolios(assets : dict, test_ret):

    """
    This function basically extract the returns of the stocks 'chosen' from our models. 
    Then compute the cumulative returns series for each strategy (top 5, top 7, top 10)

    INPUT:
    - assets : dictionary with key(=name of the portfolio), value (= list with names of assets in the portfolio)
    - test_returns : series containing the returns dataframe of all assets in the index

    RETURN-- portfolios; dictionary containing returns, cumulative returns and total performance for each portfolio
    """
    
    portfolios_returns = {}
    portfolios_perf = {}

    # Calculate the portfolios performance (equal weight portfolio on the top stocks from the previous ranking)
    for key, choices in assets.items():

        returns = test_ret[choices]

        n_assets = len(choices)

        # Calculate daily returns for our portfolio
        returns = returns * 1/n_assets  # weight daily returns of each stock by equal weight
        daily_portfolio_returns = pd.Series(returns.sum(axis=1), index=returns.index) #sum over columns the weighted returns

        # store in the dictionary the series and the portfolio performance
        portfolios_returns[key + ' returns'] = daily_portfolio_returns
        portfolios_perf[key + ' total performance']  = ((1 + daily_portfolio_returns).prod() - 1) * 100
    
    return portfolios_perf, portfolios_returns


def plot_portfolios(portfolios_returns: dict, benchmark_returns, renderer=False):

    traces = []

    benchmark_perf = (1 + benchmark_returns).cumprod()
    
    traces.append(go.Scatter(x=benchmark_perf.index, y=benchmark_perf.values, mode='lines', name='SX5E performance'))
    
    for key, series in portfolios_returns.items():

        series_perf = (1 + series).cumprod()

        traces.append(go.Scatter(x=series_perf.index, y=series_perf, mode='lines', name=key))
        
    layout = go.Layout(title='Cumulative returns : Top N Portfolios vs SX5E (benchmark)',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Values'))
        
    fig = go.Figure(data=traces, layout=layout)
    
    if renderer:
        fig.show(renderer='svg')
    else:
        fig.show()

# functions for training and evaluation - training DL.ipynb
# alternative to keras' TimeSeriesGenerator
def split_sequence(sequence, look_back, forecast_horizon):
    X, y = list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = lag_end + forecast_horizon
        if forecast_end > len(sequence):
            break
        seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)