# May 25 - draft 16:for deployment
# import libraries and packages

import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
# need this to render plotly images
import plotly.io as pio
pio.renderers.default='iframe'

# -----------------------------------------------
# load and process data and create a master dataframe with all models performance

nq_moderate = pd.read_excel('Models.xlsx', sheet_name = "NQ Moderate")
nq_moderately_conservative = pd.read_excel('Models.xlsx', sheet_name = "NQ Moderately Conservative")
nq_conservative = pd.read_excel('Models.xlsx', sheet_name = "NQ Conservative")
nq_moderately_aggressive = pd.read_excel('Models.xlsx', sheet_name = "NQ Moderately Aggressive")
nq_aggressive = pd.read_excel('Models.xlsx', sheet_name = "NQ Aggressive")
q_conservative =pd.read_excel('Models.xlsx', sheet_name = "Q Conservative")
q_moderately_conservative =pd.read_excel('Models.xlsx', sheet_name = "Q Moderately Conservative")
q_moderate = pd.read_excel('Models.xlsx', sheet_name = "Q Moderate")
q_moderately_aggressive =pd.read_excel('Models.xlsx', sheet_name = "Q Moderately Aggressive")
q_aggressive =pd.read_excel('Models.xlsx', sheet_name = "Q Aggressive")

models = [nq_conservative,
         nq_moderately_conservative,
         nq_moderate,
          nq_moderately_aggressive,
         nq_aggressive,
         q_conservative,
        q_moderately_conservative,
        q_moderate,
        q_moderately_aggressive,
        q_aggressive]

master_df = pd.DataFrame()
# this model was implemented on April 29, 2021
start = '2021-04-29'

# end date will be yesterday
end = datetime.date.today() - datetime.timedelta(days =1)


for i in models:
    df = i

    # Create a list of all tickers and weights
    tickers =  df['Ticker'].tolist()
    weights = df['Weight'].tolist()

    # Download price data for all securities. We will be using Adjusted Closing price as a proxy for stock price. 0P0001I2A1.L is a money-market fund used as a proxy for cash.
    prices = yf.download(tickers = tickers, start = start, end = end, progress = False)
    prices = prices.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
    prices = prices.round(3)
    prices = prices.rename(columns ={'0P0001I2A1.L':'Cash'})
    prices= prices['Adj Close'].dropna()

    # Calculate daily percentage change in prices
    daily_returns = round(prices.pct_change(),3)[1:]

    weighted_returns = (weights * daily_returns)
    # this shows weighted average returns of the individual holdings in our portfolio

    # aggregate the weights to calculate the daily portfolio return
    portfolio_returns = weighted_returns.sum(axis = 1)

     # calculate portfolio cumulative returns
    cumulative_returns = (portfolio_returns + 1).cumprod()
    master_df = pd.concat([master_df, cumulative_returns.to_frame()],axis = 1)

master_df.columns = ['NQ Conservative','NQ Moderately Conservative','NQ Moderate',
                    'NQ Moderately Aggressive', 'NQ Aggressive', 'Q Conservative',
                    'Q Moderately Conservative', 'Q Moderate',
                    'Q Moderately Aggressive', 'Q Aggressive']

# -------------------------------------------------------

# benchmark dataframe
bench_df = pd.DataFrame()

# benchmark holdings
bench_tickers = ['^GSPC', 'AGG', '0P0001I2A1.L']
bench_weights = [[0.3,0.6,0.1], # nq conservative
                 [0.54,0.38,0.08], # nq moderately conservative
                 [0.62,0.3,0.08], # nq moderate
                [0.7,0.22,0.08], # nq moderately aggressive
                [0.87,0.05,0.08], # nq aggressive
                [0.3,0.6,0.1], # q conservative
                [0.54,0.38,0.08], # q moderately conservative
                [0.62,0.3,0.08], # q moderate
                [0.7,0.22,0.08], # q moderately aggressive
                [0.86,0.06,0.08]] # q aggressive

# download benchmark price data
bench_prices = yf.download(tickers = bench_tickers, start = start, end = end, progress = False)
bench_prices = bench_prices.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
bench_prices = bench_prices.round(3)
bench_prices = bench_prices.rename(columns ={'0P0001I2A1.L':'Cash'})
bench_prices = bench_prices['Adj Close'].dropna()

bench_daily_returns = round(bench_prices.pct_change(),3)[1:]
# this shows daily returns for the benchmark components

for i in range(len(bench_weights)):
    bench_weighted_returns = (bench_weights[i] * bench_daily_returns)
    # this shows weighted average returns of the individual holdings in the benchmark

    # aggregate the weights to calculate the daily benchmark return
    benchmark_returns = bench_weighted_returns.sum(axis = 1)

    # calculate benchmark cumulative returns
    bench_cumulative_returns = (benchmark_returns + 1).cumprod()
    bench_df = pd.concat([bench_df, bench_cumulative_returns.to_frame()],axis=1)

bench_df.columns = ['NQ Conservative','NQ Moderately Conservative','NQ Moderate',
                   'NQ Moderately Aggressive', 'NQ Aggressive', 'Q Conservative',
                    'Q Moderately Conservative', 'Q Moderate',
                    'Q Moderately Aggressive', 'Q Aggressive']

# ---------------------------------------------------

# download S&P price data
sp = yf.download(tickers = '^GSPC', start = start, end = end, progress = False)
sp = sp.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
sp = sp.round(3)
sp = sp['Adj Close'].dropna()

# this shows daily returns for S&P
sp_daily_returns = round(sp.pct_change(),3)[1:]

sp_cumulative = (sp_daily_returns + 1).cumprod()
sp_cumulative

sp_df = pd.DataFrame()
sp_df = pd.concat([sp_df,sp_cumulative.to_frame()],axis=1)

sp_df.columns = ['S&P 500']

# -----------------------------------------------

# individual security performance

individual_df = pd.DataFrame()

for i in models:
    df = i

    # Create a list of all tickers and weights
    tickers =  df['Ticker'].tolist()
    weights = df['Weight'].tolist()

    # Download price data for all securities. We will be using Adjusted Closing price as a proxy for stock price. 0P0001I2A1.L is a money-market fund used as a proxy for cash.
    prices = yf.download(tickers = tickers, start = start, end = end, progress = False)
    prices = prices.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
    prices = prices.round(3)
    prices = prices.rename(columns ={'0P0001I2A1.L':'Cash'})
    prices= prices['Adj Close'].dropna()

    # Calculate daily percentage change in prices
    daily_returns = round(prices.pct_change(),3)[1:]

    # calculate individual stock performance
    individual_cumulative_returns = round((daily_returns+1).cumprod(),3)
    individual_df = pd.concat([individual_df,individual_cumulative_returns], axis = 1)

individual_df = individual_df.loc[:,~individual_df.columns.duplicated()]

a = individual_df.filter(items=models[0]['Ticker'])
b = individual_df.filter(items=models[1]['Ticker'])
c = individual_df.filter(items=models[2]['Ticker'])
d = individual_df.filter(items=models[3]['Ticker'])
e = individual_df.filter(items=models[4]['Ticker'])
f = individual_df.filter(items=models[5]['Ticker'])
g = individual_df.filter(items=models[6]['Ticker'])
h = individual_df.filter(items=models[7]['Ticker'])
i = individual_df.filter(items=models[8]['Ticker'])
j = individual_df.filter(items=models[9]['Ticker'])

dict = {
    'NQ Conservative' : a,
    'NQ Moderately Conservative' : b,
    'NQ Moderate' : c,
    'NQ Moderately Aggressive' : d,
    'NQ Aggressive' : e,
    'Q Conservative' : f,
    'Q Moderately Conservative' : g,
    'Q Moderate' : h,
    'Q Moderately Aggressive' :  i,
    'Q Aggressive' : j
}

# -----------------------------------------------

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

# -----------------------------------------------

# Build App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "sababaportfolio"

# ----------------------------------------------

# App Layout

app.layout = html.Div(
    children=[
        html.Div(
            children=[

        # header
                html.P(children="📈", className="header-emoji"),
                html.H1(
                    children="Portfolio Performance", className="header-title"
                ),
                html.P(
                    children="Analyze the performance of portfolio models"
                    " and their individual components against a weighted"
                    " benchmark and the S&P500 since implementation date",
                    className="header-description",
                ),
            ],
            className="header",
        ),

        # dropdown menu
        html.Div([
        dcc.Dropdown(
        id='model_dropdown',
        options=[
            {'label': 'NQ Conservative', 'value': 'NQ Conservative'},
            {'label': 'NQ Moderately Conservative', 'value': 'NQ Moderately Conservative'},
            {'label': 'NQ Moderate', 'value': 'NQ Moderate'},
            {'label': 'NQ Moderately Aggressive', 'value': 'NQ Moderately Aggressive'},
            {'label': 'NQ Aggressive', 'value': 'NQ Aggressive'},
            {'label': 'Q Conservative', 'value': 'Q Conservative'},
            {'label': 'Q Moderately Conservative', 'value': 'Q Moderately Conservative'},
            {'label': 'Q Moderate', 'value': 'Q Moderate'},
            {'label': 'Q Moderately Aggressive', 'value': 'Q Moderately Aggressive'},
            {'label': 'Q Aggressive', 'value': 'Q Aggressive'}
        ],
        value='NQ Moderate',
        className='container',
        ),
        html.Div(id='dd-output-container')
]),

        # graphs
        dcc.Graph(
            id = 'portfolio_graph',
            config={"displayModeBar": False},
            className="card"
        ),
        dcc.Graph(
            id = 'individual_graph',
            config={"displayModeBar": False},
            className="card"
        ),
    ]
)

# ---------------------------------------------------------

# portfolio graph

@app.callback(
    Output(component_id = 'portfolio_graph', component_property='figure'),
    [Input(component_id = 'model_dropdown', component_property='value')]
)


def update_portfolio_graph(column_chosen):
    dff = master_df
    dfg = bench_df
    dfh = sp_df

    # create a graph of portfolio performance
    fig=px.line()
    fig.add_scatter(x = dff.index, y = dff[column_chosen], mode = 'lines', name = "Portfolio",)
    fig.add_scatter(x = dff.index, y = dfg[column_chosen], mode = 'lines', name = "Benchmark")
    fig.add_scatter(x = dff.index, y = dfh['S&P 500'], mode = 'lines', name = "S&P 500")
    fig.update_layout(
        title={
            'text': "Portfolio Performance vs. Benchmark and S&P500, Cumulative Return on $1",
            'y' : 0.95,
            'x' : 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig

# ---------------------------------------------------------

# individual security graph

@app.callback(
    Output(component_id = 'individual_graph', component_property='figure'),
    [Input(component_id = 'model_dropdown', component_property='value')]
)

def update_individual_graph(column_chosen):
    fig2 = px.line(dict[column_chosen], labels={
                 "value" : "Cumulative Return on $1",
                 "variable" : "Security"})
    fig2.update_layout(
        title={
            'text': "Individual Security Performance",
            'y' : 0.95,
            'x' : 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig2

# ------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
