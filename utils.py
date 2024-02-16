import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


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
        result = (df - df.shift(1) / df.shift(1))
    if drop_na:
        result = result.dropna()
    return result


def cumulative_returns_from_series(series: pd.Series, log=False, start=1) -> np.array:
    """
    :param series: pandas series of returns.
    :param log: True if returns are in logarithmic form.
    :param start: starting value of returns (starting price). Default=1.
    :return: array of cumulative returns.
    """
    result = []
    for i in range(len(series)):
        if i == 0:
            result.append(start)
        else:
            if not log:
                # Compute standard cumulative returns
                ret_i = result[-1] * (1 + series.iloc[i])
                result.append(ret_i)
            else:
                # Compute logarithmic cumulative returns
                ret_i = np.exp(series.iloc[i]) * result[-1]
    return np.array(result)


def cumulative_returns(df: pd.DataFrame, log=False, start: list = None) -> pd.DataFrame:
    """
    :param df: starting pandas dataframe with returns
    :param log: True if returns are in logarithmic form.
    :param start: strting prices
    :return: dataframe with cumulative returns
    """
    if start is None:
        start = [1 for i in range(df.shape[1])]
    else:
        if len(start) != df.shape[1]:
            sys.exit("start parameter not valid. Length must be {}".format(df.shape[1]))
    result = dict()
    # Compute cum-returns for each column (stock)
    for i in range(len(start)):
        result[df.columns[i]] = cumulative_returns_from_series(df[df.columns[i]], log=log, start=start[i])
    return pd.DataFrame(result)


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
        portfolios[f'Top{i}'] = list(ranking[:i])

    # Return 4 lists with names of the top stocks according to the model's ranking
    # basically the stocks composing each portfolio with N stocks
       
    return portfolios


def calc_portfolios(assets : dict, test_ret):

    perf_portfolios = {}
    portfolios_series = {}

    # Calculate the portfolios performance (equal weight portfolio on the top stocks from the previous ranking)
    for key, value in assets.items():

        n_assets = len(value)

        cum_test =  (1 + test_ret[value]).prod() - 1

        # calculate equal weight performance of our portfolio 
        tot_performance = sum(cum_test * (1/n_assets))

        # Calculate daily returns for our portfolio
        test_ret *= 1/n_assets  #weight daily returns of each stock by equal weight
        daily_portfolio_returns = pd.Series(test_ret.sum(axis=1), index=test_ret.index) #sum over columns the weighted returns

        # store in the dictionary
        perf_portfolios[key + ' performance'] = tot_performance
        portfolios_series[key + ' series'] = daily_portfolio_returns

    return perf_portfolios, portfolios_series
    

def plot_portfolios(portfolios_series: dict, index_ret, title=None):

    index_perf = (1 + index_ret).cumprod()

    plt.figure(figsize=(16, 9))
    plt.plot(index_perf, label = 'SX5E performance')

    for key, value in portfolios_series.items():

        portfolio_perf = (1 + value).cumprod()

        plt.plot(portfolio_perf, label = key)

    plt.title(title) 
    plt.legend()
    plt.show()

    


    


    
