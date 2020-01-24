import pytz
import pyarrow
import requests
import time
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

from google.cloud import bigquery
from google.cloud import storage
from pymail import emailserver

import alpaca_trade_api as tradeapi

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# CLOUD STORAGE DEPENDENCIES & CREDENTIALS
storage_client = storage.Client()
bucket = storage_client.get_bucket('neal-trading-bot-bucket')
td_key = bucket.blob('tdapikey').download_as_string()
alpaca_key_public = bucket.blob('alpaca_creds.txt').download_as_string().decode().split(',')[0]
alpaca_key_secret = bucket.blob('alpaca_creds.txt').download_as_string().decode().split(',')[1]
bq_client = bigquery.Client()
server = emailserver()

email_list = []
# take care of concealing emails from people on github :)
with open('.email_list') as email_file:
        csv_emails = email_file.read()
        email_list = csv_emails.split(',')

# CURRENT DATE
today = datetime.today().astimezone(pytz.timezone("America/New_York"))
today_fmt = today.strftime('%Y-%m-%d')

# RESTful interface URLs
td_url = 'https://api.tdameritrade.com/v1/marketdata/EQUITY/hours'
alpaca_url = "https://paper-api.alpaca.markets"

# ALGORITHM CONFIGURATION
momentum_window = 125 # max days in algo calculation
minimum_momentum = 40 # min momentum window to be valid
portfolio_size = 10 # desired portfolio size

# Prime the API Object & RESTful interfaces
alpaca_api = tradeapi.REST(
        alpaca_key_public,
        alpaca_key_secret,
        alpaca_url,
        'v2')
positions = alpaca_api.list_positions()
account = alpaca_api.get_account()
alpaca_clock = alpaca_api.get_clock()

# SQL Queries for bot
sql_price_hist_query = """
    SELECT
      symbol,
      closePrice,
      date
    FROM 
      `neal-trading-bot.equity_data.running_quote_data`
    """

def load_portfolio():
    symbol, qty, market_value = [], [], []
    for item in positions:
        symbol.append(item.symbol)
        qty.append(int(item.qty))
        market_value.append(float(item.market_value))
    return pd.DataFrame({
            'symbol': symbol,
            'qty': qty,
            'market_value': market_value})

def query_database_for_mkt_history():
    df_mkt_hist = bq_client.query(sql_price_hist_query).to_dataframe()
    df_mkt_hist['date'] = pd.to_datetime(df_mkt_hist['date'])
    df_mkt_hist = df_mkt_hist.sort_values(by='date').reset_index(drop=True)
    
    # Get today's date since the scraper runs daily
    current_data_date = df_mkt_hist['date'].max()
    df_mkt_hist = df_mkt_hist.rename(columns={'closePrice': 'close'})
    
    # protects against double scraping - alternative: could use DISTINCT in SQL Query
    df_mkt_hist.drop_duplicates(subset=['symbol', 'date'], inplace=True)
    # get rid of penny stocks by our definition NOT SEC definition
    df_mkt_hist.drop(df_mkt_hist[df_mkt_hist.close < 2.5].index, inplace=True)
    
    return df_mkt_hist, current_data_date

# Make trades through the AlpacaAPI Wrapper
def make_trade(df, side):
    symbols = df['symbol'].tolist()
    qtys = df['delta'].tolist()
    for symbol, qty in list(zip(symbols, qtys)):
        try:
            alpaca_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day')
            print('submitted an order to {} {} shares of ${}.'.format(side, qty, symbol))
        except Exception as e:
            print('{} Exception: Could not buy ${}.'.format(side, symbol))
            print(e)
            server.send(email_list[1], # owner email i.e. me
                    '{} EXCEPTION'.format(side.upper()), 
                    'Could not {} ${}.\n\n{}'.format(side, symbol, e))
            pass

# Custom Annualizer Function
def annualize(current, initial, delta):
    d_pf = current - initial
    return (((current / initial) ** (1/delta)) - 1)* 100, ((d_pf / abs(d_pf) * current / initial) - 1) * 100

# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)

def get_momentum_stocks(df, date, portfolio_size, cash):
    # Convert dates and filter all momentum scores to include only top `portfolio_size` movers
    df_top_movers = df.loc[df['date'] == pd.to_datetime(date)]
    df_top_movers = df_top_movers.sort_values(by='momentum', ascending=False).head(portfolio_size)

    # Create a universe of top momentum stocks
    universe = df_top_movers['symbol'].tolist()

    # Create universe as DF for these stocks
    df_universe_top_movers = df.loc[df['symbol'].isin(universe)]

    # Create pre-optimzed portfolio
    df_universe_top_movers = df_universe_top_movers.pivot_table(
        index='date', 
        columns='symbol',
        values='close',
        aggfunc='sum')

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df_universe_top_movers)
    S = risk_models.sample_cov(df_universe_top_movers)

    # Optimize by Sharpe Ratio
    ef = EfficientFrontier(mu, S, gamma=1) # Use regularization (gamma=1)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Allocate
    latest_prices = get_latest_prices(df_universe_top_movers)

    allocated = DiscreteAllocation(
        cleaned_weights,
        latest_prices,
        total_portfolio_value=cash)

    allocation = allocated.lp_portfolio()[0]

    # Put the stocks and the number of shares from the portfolio into a df
    symbols = []
    num_shares = []

    for sym, shares in allocation.items():
        symbols.append(sym)
        num_shares.append(shares)

    # Create the to-buy dataframe
    df_buy = df.loc[df['symbol'].isin(symbols)]

    # Filter out irrelevant dates
    df_buy = df_buy.loc[df_buy['date'] == date].sort_values(by='symbol')

    # Add quantity allocations into dataframe
    df_buy['qty'] = num_shares # has thrown -> ValueError

    # Calculate the new/desired equity for each stock
    df_buy['equity'] = df_buy['close'] * df_buy['qty']
    df_buy = df_buy.loc[df_buy['qty'] != 0]

    return df_buy

def compose_email(who, df_buy_order, df_sell_order, df_buy, df_portfolio, df_no_order, portfolio_value):
    email_timestamp = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    subject = '{} - nealdotpy portfolio update [automated]'.format(email_timestamp)
    #portfolio_value = round(df_portfolio['market_value'].sum(), 2)
    email_string = 'YESTERDAY\'S PORTFOLIO: equity=${}'.format(portfolio_value)
    df_new_portfolio = df_buy # df_buy represents holdings that we WANT in the next trade day... poorly named :/
    new_portfolio_value = account.equity # this is updated because buy already happened before sending email # round(df_buy['equity'].sum(), 2)

    # show yesterday's portfolio
    if df_portfolio.empty is False:
        for index, row in df_portfolio.iterrows():
            append = '${} \t {} shares \t ${}'.format(row['symbol'], row['qty'], round(row['market_value'],2))
            email_string += '\n{}'.format(append)
    else:
        email_string += '\nNo exisiting portfolio for this account (this usually occurs due to a reset).'

    email_string += '\n\nBUYS:'
    # show the buy orders that were just executed
    if df_buy_order.empty is False:
        for index, row in df_buy_order.iterrows():
            append = 'purchased {} shares of ${} for ~${}'.format(row['delta'], row['symbol'], round(row['equity'],2))
            email_string += '\n{}'.format(append)
    else:
        email_string += '\nNo buys were required today (this is rare).'

    email_string += '\n\nSELLS:'
    # show the sell orders that were just executed
    if df_sell_order.empty is False:
        for index, row in df_sell_order.iterrows():
            append = 'sold {} shares of ${} for ~${}'.format(row['delta'], row['symbol'], round(row['equity'],2))
            email_string += '\n{}'.format(append)
    else:
        email_string += '\nNo sells were required today (this is rare).'
    email_string += '\n\nNO CHANGES:'
    # show no changers
    if df_no_order.empty is False:
        for index, row in df_no_order.iterrows():
            append = 'No change made to position: ${} (${})'.format(row['symbol'], round(row['equity'],2))
            email_string += '\n{}'.format(append)
    else:
        email_string += '\nEverything changed in some way in your portfolio.'

    email_string += '\n\nTODAY\'S PORTFOLIO: equity=${}'.format(new_portfolio_value)
    # show the new portfolio
    if df_new_portfolio.empty is False:
        for index, row in df_new_portfolio.iterrows():
            append = '${} \t {} shares \t ${}'.format(row['symbol'], row['qty'], round(row['equity'],2))
            email_string += '\n{}'.format(append)
    else:
        email_string += '\nYou should never see this message. If you do, there was a critical error.\n'

    server.send(who, subject, email_string)

def trade_bot(event, context): # CHANGE SECOND LINE HERE FOR NEW CODE WITH ACCOUNT.EQUITY VVVVVVVVVVVVVV
    # Get some basic portfolio information
    df_portfolio = load_portfolio()
    portfolio_value = float(account.equity) # -> used in trade ai
    print('PORTFOLIO:\n{}\nPORTFOLIO_EQUITY_VALUE: {}'.format(df_portfolio, portfolio_value))

    # We only want to be putting our daily order in right after market opens and only IF it is open
    if alpaca_clock.is_open is not True:
        return_message = 'returned without trading. market is not open.'
        server.send(email_list[1], 'ALERT: trade.ai', 'trade.ai {}'.format(return_message))
        #return return_message

    if portfolio_value == 0: # for unit tests only
        portfolio_value = 12000
        server.send(email_list[1], 'WARNING: trade.ai used default portfolio value', 'value used {}'.format(portfolio_value))

    df_mkt_hist, current_data_date = query_database_for_mkt_history() # SQL + GCP BQ Tables

    # Take the historical database DF and append a momentum score to each symbol
    df_mkt_hist['momentum'] = df_mkt_hist.groupby('symbol')['close'].rolling(
        momentum_window,
        min_periods=minimum_momentum).apply(momentum_score, raw=True).reset_index(level=0, drop=True)

    # Call the function
    df_buy = get_momentum_stocks(
            df=df_mkt_hist,
            date=current_data_date,
            portfolio_size=portfolio_size,
            cash=portfolio_value)

    df_combined = pd.merge(df_portfolio, df_buy, on='symbol', how='outer').fillna(0)
    df_combined = df_combined.rename(columns={'qty_x': 'have', 'qty_y': 'want'})

    df_combined['delta'] = df_combined['want'] - df_combined['have']

    # clean up the df and create buy/sell/no-change dfs
    rmv = ['want', 'have', 'market_value', 'close', 'date', 'momentum'] # 'equity'
    df_buy_order = df_combined[df_combined.delta > 0].drop(rmv, axis=1)
    df_sell_order = df_combined[df_combined.delta < 0].drop(rmv, axis=1)
    df_no_order = df_combined[df_combined.delta == 0].drop(rmv, axis=1)

    # make negative deltas positive to pass into the sell API as "sell positive shares"
    df_sell_order['delta'] = abs(df_sell_order['delta'])

    if df_buy_order.empty is False:
        make_trade(df_buy_order, 'buy')
    if df_sell_order.empty is False:
        make_trade(df_sell_order, 'sell')

    compose_email(email_list[1], df_buy_order, df_sell_order, df_buy, df_portfolio, df_no_order, portfolio_value) # ME
    compose_email(email_list[0], df_buy_order, df_sell_order, df_buy, df_portfolio, df_no_order, portfolio_value) # SHIV
    compose_email(email_list[2], df_buy_order, df_sell_order, df_buy, df_portfolio, df_no_order, portfolio_value) # TERRY

    return 'Success!'

print(trade_bot(None, None)) # local testing only
#print(unit_test()) 