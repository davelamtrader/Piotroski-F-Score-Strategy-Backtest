# 1. Get Historical Data of 50 stocks
import yfinance as yf
import pandas as pd


def get_historical_data(tickers, time_loop_back='4y', interval='1mo'):
    """
    Retrieve historical price data and financial data for a list of tickers.

    Parameters:
    tickers (list): List of ticker symbols.
    time_loop_back (str): Period to look back for historical data (default is '4y').
    interval (str): Interval for historical data (default is '1mo').

    Returns:
    tuple: A tuple containing two dictionaries:
        - price_data: Dictionary with historical price data for each ticker.
        - financial_data: Dictionary with financial data (financials, cashflow, balance_sheet) for each ticker.
    """
    import yfinance as yf
    import pandas as pd

    # Create a dictionary to store historical data
    price_data = {}

    # Create a dictionary to store financial data (Financials, Cashflow, and Balance Sheet)
    financial_data = {}

    # Create a list to store tickers for which data is not available
    unavailable_tickers = []

    # Retrieve data for each ticker
    for ticker in tickers:
        # Get stock data
        stock = yf.Ticker(ticker)

        # Get Historical Price Data
        hist = stock.history(period=time_loop_back, interval=interval)
        price_data[ticker] = hist

        # Get historical Financials Data, Cashflow Data, and Balance Sheet Data
        financials = stock.financials
        cashflow = stock.cashflow
        balance_sheet = stock.balance_sheet

        # If no data is returned, add the ticker to the list of unavailable tickers
        if hist.empty:
            unavailable_tickers.append(ticker)
            continue

        # handle not matching data 
        if financials.shape[1] != cashflow.shape[1] or financials.shape[1] != balance_sheet.shape[1]:
            # find the minimum shape among financials, cashflow, and balance sheet
            min_shape = min(financials.shape[1], cashflow.shape[1], balance_sheet.shape[1])
            # trim the data to match the minimum shape
            financials = financials.iloc[:, :min_shape]
            cashflow = cashflow.iloc[:, :min_shape]
            balance_sheet = balance_sheet.iloc[:, :min_shape]

        financial_data[ticker] = {
            'financials': financials,
            'cashflow': cashflow,
            'balance_sheet': balance_sheet
        }

        print('Got data for', ticker)

    # print the tickers for which data is not available
    if unavailable_tickers:
        print('Data not available for:', unavailable_tickers)

    return price_data, financial_data

#2. Function to Calculate Piotroski F-Score
def calculate_piotroski_f_score(financials, cashflow, balance_sheet):
    """
    Calculate the Piotroski F-Score for a given company's financial data.

    Parameters:
    financials (pd.DataFrame): The financials data.
    cashflow (pd.DataFrame): The cashflow data.
    balance_sheet (pd.DataFrame): The balance sheet data.

    Returns:
    pd.Series: The Piotroski F-Score for each period.
    """

    # 1. Positive net income
    positive_net_income = financials.loc['Net Income'] > 0

    # 2. Positive operating cash flow
    positive_operating_cashflow = cashflow.loc['Operating Cash Flow'] > 0

    # 3. Positive return on assets (ROA)
    net_income = financials.loc['Net Income']
    total_assets = balance_sheet.loc['Total Assets']
    average_assets = (total_assets + total_assets.shift(1)) / 2
    return_on_assets = net_income / average_assets
    positive_roa = return_on_assets > 0

    # 4. Operating cash flow > Net income (đồng bộ nhãn trước khi so sánh)
    net_income_aligned = net_income.reindex(cashflow.columns)
    operating_cash_flow_greater_than_net_income = cashflow.loc['Operating Cash Flow'] > net_income_aligned

    # 5. Increasing ROA
    increasing_roa = return_on_assets.diff() > 0

    # 6. Positive change in long-term debt ratio
    long_term_debt = balance_sheet.loc['Long Term Debt']
    previous_long_term_debt = long_term_debt.shift(1)
    positive_change_long_term_debt_ratio = (long_term_debt < previous_long_term_debt)

    # 7. Increasing current ratio
    # current_assets = balance_sheet.loc['Total Current Assets']
    current_assets = balance_sheet.loc['Current Assets']
    # current_liabilities = balance_sheet.loc['Total Current Liabilities']
    current_liabilities = balance_sheet.loc['Current Liabilities']
    previous_current_assets = current_assets.shift(1)
    previous_current_liabilities = current_liabilities.shift(1)
    increasing_current_ratio = (current_assets / current_liabilities) > (
                previous_current_assets / previous_current_liabilities)

    # 8. No new shares issued
    new_shares_issued = balance_sheet.loc['Issuance of Capital Stock',
                        :] if 'Issuance of Capital Stock' in balance_sheet.index else pd.Series(
        [0] * len(balance_sheet.columns), index=balance_sheet.columns)
    no_new_shares_issued = new_shares_issued == 0

    # 9. Higher gross margin
    gross_margin = (financials.loc['Gross Profit'] / financials.loc['Total Revenue'])
    previous_gross_margin = gross_margin.shift(1)
    higher_gross_margin = gross_margin > previous_gross_margin

    # Calculate Piotroski F-Score
    piotroski_f_score = positive_net_income.astype(int) + positive_operating_cashflow.astype(int) + positive_roa.astype(
        int) + operating_cash_flow_greater_than_net_income.astype(int) + increasing_roa.astype(
        int) + positive_change_long_term_debt_ratio.astype(int) + increasing_current_ratio.astype(
        int) + no_new_shares_issued.astype(int) + higher_gross_margin.astype(int)

    # drop NaN values
    piotroski_f_score = piotroski_f_score.dropna()

    return piotroski_f_score

#3. Create DataFrame Containing Piotroski F-Score for All Stocks
def create_piotroski_f_score_dataframe(tickers, financial_data):
    """
    Create a DataFrame containing the Piotroski F-Score for all given tickers.

    Parameters:
    tickers (list): List of ticker symbols.
    financial_data (dict): Dictionary containing financial data for each ticker.

    Returns:
    pd.DataFrame: DataFrame containing the Piotroski F-Score for each ticker.
    """

    # Create a dictionary to store Piotroski F-Scores
    piotroski_f_scores = {}

    # Calculate Piotroski F-Score for each ticker
    for ticker in tickers:
        financials = financial_data[ticker]['financials']
        cashflow = financial_data[ticker]['cashflow']
        balance_sheet = financial_data[ticker]['balance_sheet']

        piotroski_f_score = calculate_piotroski_f_score(financials, cashflow, balance_sheet)
        piotroski_f_scores[ticker] = piotroski_f_score

    # Convert the dictionary to a DataFrame
    piotroski_f_score_df = pd.DataFrame(piotroski_f_scores)

    return piotroski_f_score_df

#4. Create DataFrame to Store Profit and Loss of Each Stock in Each Year
# ignore the warning
import warnings
warnings.filterwarnings('ignore')

# use with price_data
# only keep row month = 1 of each year
price_data_month_1 = {}
for ticker in tickers:
    price_data_month_1[ticker] = price_data[ticker][price_data[ticker].index.month == 1]

# rewrite index to keep only year
for ticker in tickers:
    price_data_month_1[ticker].index = price_data_month_1[ticker].index.year

# loop through each ticker to calculate the return of price_data_month_1
for ticker in tickers:
    price_data_month_1[ticker]['Return_next_1_year'] = price_data_month_1[ticker]['Close'].pct_change()

# display(price_data_month_1['AAPL'].head())

# shift the return_next_1_year +2
for ticker in tickers:
    price_data_month_1[ticker]['Return_next_1_year'] = price_data_month_1[ticker]['Return_next_1_year'].shift(-2)

# drop 1st & last row ( only keep data corresponding with DataFrame of top 10 stocks with Piotroski F-Score)
for ticker in tickers:
    price_data_month_1[ticker] = price_data_month_1[ticker].iloc[1:-2]

# display shape of price_data_month_1
display(price_data_month_1['AAPL'].head())
display(price_data_month_1['AMZN'].head())

#5. DataFrame to Backtest Strategy
def calculate_returns_of_strategy(df_top_10, price_data, n_top_stocks):
    df_backtest = df_top_10.copy()

    # map return of each stock to the DataFrame of top 10 stocks
    for index, row in df_top_10.iterrows():
        # get the year
        year = index.year
        # loop through the top 10 stocks
        for i in range(1, n_top_stocks+1):
            # get the ticker
            ticker = row[f'Top_{i}_Stock']
            # get the return of the stock in the next year
            return_next_1_year = price_data[ticker].loc[year, 'Return_next_1_year']
            # update the DataFrame
            df_backtest.at[index, f'Top_{i}_Return'] = return_next_1_year

    # calculate the average return of the top 10 stocks
    df_backtest['Average_Return'] = df_backtest[[f'Top_{i}_Return' for i in range(1, n_top_stocks+1)]].mean(axis=1)

    # calculate the cumulative return
    df_backtest['Cumulative_Return'] = (1 + df_backtest['Average_Return']).cumprod()

    # print final cumulative return
    print('*'*50)
    print('Pick the top', n_top_stocks, 'stocks based on Piotroski F-Score')
    print('Final Cumulative Return:', df_backtest['Cumulative_Return'].iloc[-1] * 100 - 100, '%')
    print('*'*50)

    # metrics
    metrics= {
        'Cumulative_Return_%': df_backtest['Cumulative_Return'].iloc[-1] * 100 - 100,
        'Average_Return_%': df_backtest['Average_Return'].mean() * 100,
        'Standard_Deviation_%': df_backtest['Average_Return'].std() * 100,
        'Sharpe_Ratio': df_backtest['Average_Return'].mean() / df_backtest['Average_Return'].std()
    }

    return df_backtest, metrics

df_backtest,metrics = calculate_returns_of_strategy(df_top_10_piostroski_f_score, price_data_month_1, n_top_stocks)
display(metrics)

# define the range of top stocks
range_of_n_top_stocks = range(1, 11)

# run the backtest for each range of top stocks
results = []
for _ in range_of_n_top_stocks:
    temp_df_backtest, temp_metrics = calculate_returns_of_strategy(df_top_10_piostroski_f_score, price_data_month_1, _)
    results.append(temp_metrics)

# create a DataFrame from the results
df_results = pd.DataFrame(results, index=range_of_n_top_stocks)

# create columns containing the range of top stocks
df_results['Pick_Top_Stocks'] = df_results.index

# sort the DataFrame by Sharpe Ratio
df_results = df_results.sort_values(by='Sharpe_Ratio', ascending=False)

# put the 'Pick_Top_Stocks' column in the first position
cols = df_results.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_results = df_results[cols]

# display the results
display(df_results)

import matplotlib.pyplot as plt
import seaborn as sns

# plot the results ( cumulative return with number of top stocks)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x='Pick_Top_Stocks', y='Cumulative_Return_%', marker='o')
plt.title('Cumulative Return vs Number of Top Stocks')
plt.xlabel('Number of Top Stocks')
plt.ylabel('Cumulative Return (%)')
plt.grid(True)
plt.show()


# plot the results ( sharpe ratio with number of top stocks)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x='Pick_Top_Stocks', y='Sharpe_Ratio', marker='o')
plt.title('Sharpe Ratio vs Number of Top Stocks')
plt.xlabel('Number of Top Stocks')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.show()
