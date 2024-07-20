# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinancetool as yt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2

import warnings
warnings.filterwarnings('ignore')
#global constant
Buy = 1
Sell = -1
Neutral = 0

# %%

basepath = r"C:/UH/DataScienceProject/data/"
yft = yt.YFinanceHelper(basepath)


    

# %%
# Function to calculate Weighted Moving Average (WMA)
def calculate_wma(prices, window):
    weights = np.arange(1, window + 1)
    wma = prices.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(prices, window):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_fibonacci_levels(prices, window):

    high_price = prices.rolling(window).max()
    low_price = prices.rolling(window).min()
    difference = high_price - low_price
    level1 = high_price - difference * 0.236
    level2 = high_price - difference * 0.382
    level3 = high_price - difference * 0.618
    return level1, level2, level3

def calculate_bollinger_bands(prices, window, num_of_std=2):
    SMA= prices.rolling(window=window).mean()
    STD = prices.rolling(window=window).std()
    UpperBand = SMA + (STD * num_of_std)
    LowerBand = SMA - (STD * num_of_std)
    return UpperBand,LowerBand

def calculate_percentage_change(prices, window):

    pc = []
    
    for i in range(len(prices)):
        if i < window:
            pc.append(pd.NA)
        else:
            start_price = prices[i - window]
            end_price = prices[i]
            pc.append((end_price - start_price) / start_price * 100)
    return pc

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_vpt(df, ema_period=20):
    df['PFI'] = df['Volume'] * (df['Adj Close'].diff() / df['Adj Close'].shift(1))
    df['VPT'] = df['PFI'].cumsum()
    df['VPT_EMA'] = calculate_ema(df['VPT'], ema_period)
    return df

# %%
def apply_techinal_indicators(df):

    df['WMA'] = calculate_wma(df['Adj Close'], 28)

    df['RSI'] = calculate_rsi(df['Adj Close'], 28)

    df['BUB'],df['BLB'] = calculate_bollinger_bands(df['Adj Close'], 28)

    df['PC'] = calculate_percentage_change(df['Adj Close'], 28)

    df['VPT'] = ((df['Adj Close'].diff() / df['Adj Close'].shift(1)) * df['Volume']).cumsum()
    #df['PFI'] = df['Volume'] * (df['Adj Close'].diff() / df['Adj Close'].shift(1))
    #df['VPT'] = df['PFI'].cumsum()
    df['VPT_EMA'] = calculate_ema(df['VPT'], 28)
    return df

#level1,level2,level3 = calculate_fibonacci_levels(new_df['Adj Close'], 14)
#new_df['Fib 23.6%'] = level1
#new_df['Fib 38.2%'] = level2
#new_df['Fib 61.8%'] = level3


# %%
# Step 8: Generate buy, sell, and neutral signals for WMA
def calcualte_signal_by_techinal_indicators(df):
    # df['WMA Signal'] = np.where(df['Adj Close'] < df['WMA'], 1,
    #                                 np.where(df['Adj Close'] > df['WMA'], -1, 0))
    df['WMA Signal'] = np.where(df['Adj Close'] < df['WMA'], 1,
                                    np.where(df['Adj Close'] > df['WMA'], -1, 0))
     
    # Step 9: Generate buy, sell, and neutral signals for RSI
    df['RSI Signal'] = np.where(df['RSI'] < 30, 1,
                                    np.where(df['RSI'] > 70, -1, 0))

      
    # Step 9: Generate buy, sell, and neutral signals for RSI

    df['PC Signal'] = np.where(df['PC'] <= -8, 1,
                                    np.where(df['PC'] >= 8, -1, 0))#-5-15


    df['BB Signal'] = np.where(df['Adj Close'] < df['BLB'], 1,
                                    np.where(df['Adj Close'] > df['BUB'], -1, 0))
    
    
    # df['VPT Signal'] = 0
    # df.loc[df['VPT'].diff() > 0, 'VPT Signal'] = -1
    # df.loc[df['VPT'].diff() < 0, 'VPT Signal'] = 1

    df['VPT Signal'] = 0
    df.loc[df['VPT'] > df['VPT_EMA'], 'VPT Signal'] = -1
    df.loc[df['VPT'] < df['VPT_EMA'], 'VPT Signal'] = 1

    # df['WMA RSI Signal'] = np.where((df['WMA Signal'] == 1) & (df['RSI Signal']  == 1) & (df['VPT Signal']  == 1), 1,
    #                                 np.where((df['WMA Signal'] == -1) & (df['RSI Signal']  == -1) & (df['VPT Signal']  == -1), -1, 0))
 
    df['WMA RSI Signal'] = np.where((df['WMA Signal'] == 1) & (df['VPT Signal']  == 1), 1,
                                    np.where((df['WMA Signal'] == -1) & (df['VPT Signal']  == -1), -1, 0))
 
    

    df.to_csv("test.csv")
    return df

##bb, parabolic sar

#new_df['All Signal'] = np.where((new_df['WMA Signal'] == 'Buy') & (new_df['RSI Signal'] == 'Buy') & (new_df['Fib Signal'] == 'Buy'), 'Buy',
#                                np.where((new_df['WMA Signal'] == 'Sell') & (new_df['RSI Signal'] == 'Sell') & (new_df['Fib Signal'] == 'Sell'), 'Sell', 'Neutral'))

#new_df['All Signal'] = np.where((new_df['WMA Signal'] == 'Buy') & (new_df['RSI Signal'] == 'Buy') & (new_df['Fib Signal'] == 'Buy'), 'Buy',
#                                np.where((new_df['WMA Signal'] == 'Sell') & (new_df['RSI Signal'] == 'Sell') & (new_df['Fib Signal'] == 'Sell'), 'Sell', 'Neutral'))


# %%
def calculate_profit(adj_close,signal, initial_cash=10000):
    cash = initial_cash
    stock = 0
    portfolio_value = []
     # Filter dataframe between entry and exit dates
    for i in range(len(adj_close)):
        if signal.iloc[i] == 1 and cash > 0:
            # Buy as many stocks as possible with available cash
            stock = cash / adj_close.iloc[i]
            cash = 0
            #print(singal,df.iloc[i]["Date"], "Buy")
        elif signal.iloc[i] == -1 and stock > 0:
            # Sell all stocks
            cash = stock * adj_close.iloc[i]
            stock = 0
            #print(singal,df.iloc[i]["Date"], "Sell")
        # Calculate the current value of the portfolio
        current_value = cash + stock * adj_close.iloc[i]
        portfolio_value.append(current_value)

    final_value = cash + stock * adj_close.iloc[-1]
    profit = final_value - initial_cash
    profit_percentage = (profit / initial_cash) * 100
  #  print(df)
    return profit, portfolio_value, profit_percentage




# %%
def plot(ticker, signal, df):
   
    fig = go.Figure()

    # Plot Adjusted Close Price
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

    if signal == "WMA Signal":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['WMA'], mode='lines', name='WMA', line=dict(color='orange')))
    elif signal == "RSI Signal":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    elif signal == "BB Signal":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BUB'], mode='lines', name='BUB', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BLB'], mode='lines', name='BLB', line=dict(color='orange')))

    df_buy_signals = df[df[signal] == 1]
    df_sell_signals = df[df[signal] == -1]
    
    fig.add_trace(go.Scatter(x=df_buy_signals['Date'], y=df_buy_signals['Adj Close'], mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['Adj Close'], mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', size=10, color='red')))
    
    fig.update_layout(
        title=f'{ticker} Stock Analysis {signal}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_dark',
        height=600,
        width=1000
    )

    fig.show()

def plot2(title, x,y,x_buy,y_buy,x_sell,y_sell):
   
    fig = go.Figure()

    # Plot Adjusted Close Price
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Adj Close', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=x_buy, y=y_buy, mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=x_sell, y=y_sell, mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', size=10, color='red')))
    
    fig.update_layout(
        title=f'{title} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_dark',
        height=600,
        width=1000
    )

    fig.show()

# %%

def plot_profit(ticker,signal,df,entry_date,exit_date):
  
    df = df[(df['Date'] >= entry_date) & (df['Date'] <= exit_date)]
    
    plt.figure(figsize=(20, 7))

    # Plot Adjusted Close Price
    plt.plot(df['Date'], df['Adj Close'], label='Adj Close', color='blue')

    if(signal=="WMA Signal"):
        plt.plot(df['Date'], df['WMA'], label='WMA', color='orange')
    if(signal=="RSI Signal"):
        plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
    
    if(signal=="BB Signal"):
        plt.plot(df['Date'], df['BUB'], label='BUB', color='orange')
        plt.plot(df['Date'], df['BLB'], label='BLB', color='orange')

    df_buy_signals = df[df[signal] == 1]
    df_sell_signals = df[df[signal] == -1]
    
    plt.scatter(df_buy_signals['Date'], df_buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df_sell_signals['Date'], df_sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    

    # Plot Fibonacci Levels
    #plt.plot(df['Date'], df['BUB'], color='r', linestyle='--', label='BUB')
    #plt.plot(df['Date'], df['BLB'], color='g', linestyle='--', label='BLB')


    plt.title(f'{ticker} Stock Analysis {signal}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
# Specify entry and exit dates
def calculate_all_profit(ticker,df):
    pc_profit, pc_portfolio_value, pc_profit_percentage  = calculate_profit(df["Adj Close"],df["PC Signal"])
    print(f"PC Signal : Final Profit: ${pc_profit:.2f}, {pc_profit_percentage.round(2)}%")
    
    wam_profit, wam_portfolio_value, wam_profit_percentage = calculate_profit(df["Adj Close"],df["WMA Signal"])
    print(f"WMA Signal : Final Profit: ${wam_profit:.2f}, {wam_profit_percentage.round(2)}%")

    rsi_profit, rsi_portfolio_value, rsi_profit_percentage = calculate_profit(df["Adj Close"],df["RSI Signal"])
    print(f"RSI Signal : Final Profit: ${rsi_profit:.2f}, {rsi_profit_percentage.round(2)}%")

    bb_profit, bb_portfolio_value, bb_profit_percentage = calculate_profit(df["Adj Close"],df["BB Signal"])
    print(f"BB Signal : Final Profit: ${bb_profit:.2f}, {bb_profit_percentage.round(2)}%")

    plot(ticker,"PC Signal",df)
    plot(ticker,"WMA Signal",df)
    plot(ticker,"RSI Signal",df)
    plot(ticker,"BB Signal",df)
    

# %%
def train_logistic_regression_model(df_train,features, target):
        
    # Create features and target
    df_features = df_train[features]
    target = df_train[target]  # Using SMA Signal as target for this example

    # # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)

    # Apply Chi-Squared feature selection
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_train, y_train)

    # Get the selected features
    selected_features = chi2_selector.get_support(indices=True)
    print(f'Selected features (CHI): {selected_features}')
    print(f'Selected features (CHI): {df_features.columns[selected_features]}')
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, scaler
   
def train_gradient_regression_model(df_train,features, target):
    #    https://stackoverflow.com/questions/56505564/handling-unbalanced-data-in-gradientboostingclassifier-using-weighted-class
    # Create features and target
    df_features = df_train[features]
    target = df_train[target]  # Using SMA Signal as target for this example
    

    # # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 0.1
    sample_weights[y_train == 1] = 0.8
    sample_weights[y_train == -1] = 0.8

    #Define the parameter grid
    #'learning_rate': 0.2, 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.8}
    param_grid = {
        'n_estimators': [500, 700, 1000], #[50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [5, 6, 7] #3, 4, 5
            }
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'max_depth': [3, 4, 5, 6],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'subsample': [0.8, 0.9, 1.0]
    # }
    # Initialize the Gradient Boosting Regressor
    gbm = GradientBoostingRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train, sample_weight = sample_weights)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f'Best parameters: {best_params}')

    # Train the model with the best parameters
    best_gbm = grid_search.best_estimator_

    # Make predictions
    y_pred = best_gbm.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error after tuning: {mse}')

    # # Train the gradient regression model
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    # # Train the model
    # model.fit(X_train, y_train)

    # # Make predictions
    # y_pred = model.predict(X_test_scaled)

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # print(f'Mean Squared Error: {mse}')
    return best_gbm, scaler
   
    


# %%
def predict_signals(model,scaler,df,features):

    df_features = df[features]
    X_test_scaled = scaler.transform(df_features)
    y_pred = model.predict(X_test_scaled)
    return y_pred



# %%
def adjust_consecutive_signals(signals):

    buy_indices = signals[signals['WMA RSI Signal Optimized'] == 1].index
    sell_indices = signals[signals['WMA RSI Signal Optimized'] == -1].index

    # Process buy signals
    signals = process_consecutive_signals(signals, buy_indices, 'max')

    # Process sell signals
    signals = process_consecutive_signals(signals, sell_indices, 'min')

    return signals

def process_consecutive_signals(signals, indices, method):

    if method == 'max':
        func = np.argmax
    elif method == 'min':
        func = np.argmin

    consecutive_groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    for group in consecutive_groups:
        if len(group) > 2:
            keep_idx = group[func(signals.loc[group, 'Adj Close'])]
            signals.loc[group, 'WMA RSI Signal Optimized'] = 0
            signals.loc[keep_idx, 'WMA RSI Signal Optimized'] = 1 if method == 'max' else -1

    return signals

# %%

sd = '1887-12-31' #'2014-01-01'
ed = '2023-12-31' #'1999-12-31'#'2023-12-31'

test_exit_date = pd.to_datetime(ed)
test_entry_date = test_exit_date - pd.Timedelta(days=365*1)

train_start_date = pd.to_datetime(sd)
train_end_date = test_entry_date
#appl
ticker_symbol = "MSFT" #PFE, AAPL, MSFT


stock_df = yft.get_data(ticker_symbol,sd,ed)[['Date', 'Adj Close','Volume']]
new_df = apply_techinal_indicators(stock_df)
new_df.dropna(inplace=True)
new_df = calcualte_signal_by_techinal_indicators(new_df)
#new_df['WMA RSI Signal'].value_counts()

# %%

new_df['WMA RSI Signal Optimized'] = new_df['WMA RSI Signal']

new_df['idx'] = new_df.index
#new_df = adjust_consecutive_signals(new_df)
#new_df = adjust_consecutive_signals(new_df)
#new_df = adjust_consecutive_signals(new_df)
neutral_indexs = []
buy_sell_indices = new_df[(new_df['WMA RSI Signal'] == 1) | (new_df['WMA RSI Signal'] == -1)]
 #   sell_indices = signals[signals['WMA RSI Signal Optimized'] == -1].index
cnt = 0
group_signal=[]

for index, row in buy_sell_indices.iterrows():
    if row['WMA RSI Signal']==0:
        continue
    if cnt == 0:
        last_row = row
        last_index = index
        cnt=cnt+1
        continue

    previous_signal = last_row['WMA RSI Signal']
    previous_price = last_row['Adj Close']
    current_signal = row['WMA RSI Signal']
    current_price = row['Adj Close']
    if previous_signal == current_signal:
        if (current_signal == 1): #buy
            min_price = current_price
            group_signal.append(last_row['idx'])
            #new_df["WMA RSI Signal Optimized"].iloc[i-1] = 0
        elif (current_signal == -1) : #dell
            max_price=current_price
            group_signal.append(last_row['idx'])
            #new_df["WMA RSI Signal Optimized"].iloc[i-1] = 0
    else:
        if(len(group_signal)>0):
            if(new_df['WMA RSI Signal'][group_signal[0]]==1): #buy
                min_index = new_df.loc[new_df["idx"].isin(group_signal)]['Adj Close'].idxmin()
                new_df['WMA RSI Signal Optimized'][group_signal] = 0
                new_df['WMA RSI Signal Optimized'][min_index] = 1
            if(new_df['WMA RSI Signal'][group_signal[0]]==-1): #buy
                max_index = new_df.loc[new_df["idx"].isin(group_signal)]['Adj Close'].idxmax()
                new_df['WMA RSI Signal Optimized'][group_signal] = 0
                new_df['WMA RSI Signal Optimized'][max_index] = -1
        group_signal=[]
        
    last_row = row
    last_index = index
    cnt=cnt+1



# #new_df[new_df['idx'].isin(neutral_indexs)] = 0

# print(new_df['WMA RSI Signal Optimized'].value_counts())

# wam_rsi_op_profit, wam_rsi_op_portfolio_value, wam_rsi_op_profit_percentage = calculate_profit(new_df["Adj Close"],new_df["WMA RSI Signal Optimized"])
# print(f"WMA RSI Signal Optimized: Final Profit: ${wam_rsi_op_profit:.2f}, {wam_rsi_op_profit_percentage.round(2)}%")
# plot(ticker_symbol,"WMA RSI Signal Optimized",new_df)

# wam_rsi_profit, wam_rsi_portfolio_value, wam_rsi_profit_percentage = calculate_profit(new_df["Adj Close"],new_df["WMA RSI Signal"])
# print(f"WMA RSI Signal : Final Profit: ${wam_rsi_profit:.2f}, {wam_rsi_profit_percentage.round(2)}%")
# plot(ticker_symbol,"WMA RSI Signal",new_df)
#new_df




# %%
##### main #####

features = ['Adj Close','WMA','VPT','RSI'] #'RSI','VPT','Volume'

#train on previous year
df_train = new_df[(new_df['Date'] >= train_start_date) & (new_df['Date'] <= train_end_date)]
#print (df_train)
#clean data



model_lr,scaler_lr = train_logistic_regression_model(df_train,features,"WMA RSI Signal")
model_gbr,scaler_gbr = train_gradient_regression_model(df_train,features,"WMA RSI Signal")

#test on last 1 year data
df_test = new_df[(new_df['Date'] >= test_entry_date) & (new_df['Date'] <= test_exit_date)]
#print(df_filter)





#calculate_all_profit(ticker_symbol,df_filter)
#df_test["Logistic WMA RSI Signal"] =  predict_signals(model_lr,scaler_lr,df_test,features)
df_test["Gradient WMA RSI Signal"] =  predict_signals(model_gbr,scaler_gbr,df_test,features)

    

    #bb_profit, bb_portfolio_value, bb_profit_percentage =calculate_profit(X_test["Adj Close"], df_y_pred["Logistic Signal"])
    #print(f"Logistic Signal : Final Profit: ${bb_profit:.2f}, {bb_profit_percentage.round(2)}%")
    
# df_buy_signals = df_filter[df_filter["Logistic Signal"] == 1]
# df_sell_signals = df_filter[df_filter["Logistic Signal"] == -1]
# plot2("Logistic WMA",df_filter['Date'], df_filter['Adj Close'],df_buy_signals['Date'],df_buy_signals['Adj Close'],df_sell_signals['Date'],df_sell_signals['Adj Close'] )


# ml_wam_profit, ml_wam_portfolio_value, ml_wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["Logistic WMA RSI Signal"])
# print(f"Logistic WMA RSI Signal : Final Profit: ${ml_wam_profit:.2f}, {ml_wam_profit_percentage.round(2)}%")
# plot(ticker_symbol,"Logistic WMA RSI Signal",df_test)

ml_wam_profit, ml_wam_portfolio_value, ml_wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["Gradient WMA RSI Signal"])
print(f"Gradient WMA RSI Signal : Final Profit: ${ml_wam_profit:.2f}, {ml_wam_profit_percentage.round(2)}%")
plot(ticker_symbol,"Gradient WMA RSI Signal",df_test)

wam_rsi_op_profit, wam_rsi_op_portfolio_value, wam_rsi_op_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["WMA RSI Signal Optimized"])
print(f"WMA RSI Signal Optimized: Final Profit: ${wam_rsi_op_profit:.2f}, {wam_rsi_op_profit_percentage.round(2)}%")
plot(ticker_symbol,"WMA RSI Signal Optimized",df_test)

wam_profit, wam_portfolio_value, wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["WMA RSI Signal"])
print(f"WMA RSI Signal : Final Profit: ${wam_profit:.2f}, {wam_profit_percentage.round(2)}%")
plot(ticker_symbol,"WMA RSI Signal",df_test)

wam_profit, wam_portfolio_value, wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["WMA Signal"])
print(f"WMA Signal : Final Profit: ${wam_profit:.2f}, {wam_profit_percentage.round(2)}%")
plot(ticker_symbol,"WMA Signal",df_test)

wam_profit, wam_portfolio_value, wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["RSI Signal"])
print(f"RSI Signal : Final Profit: ${wam_profit:.2f}, {wam_profit_percentage.round(2)}%")
plot(ticker_symbol,"RSI Signal",df_test)

vpt_profit, vpt_portfolio_value, vpt_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["VPT Signal"])
print(f"VPT Signal : Final Profit: ${wam_profit:.2f}, {vpt_profit_percentage.round(2)}%")
plot(ticker_symbol,"VPT Signal",df_test)


wam_profit, wam_portfolio_value, wam_profit_percentage = calculate_profit(df_test["Adj Close"],df_test["PC Signal"])
print(f"PC Signal : Final Profit: ${wam_profit:.2f}, {wam_profit_percentage.round(2)}%")
plot(ticker_symbol,"PC Signal",df_test)

# Fitting 5 folds for each of 972 candidates, totalling 4860 fits
# Best parameters: {'learning_rate': 0.2, 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.8}
# Mean Squared Error after tuning: 0.13197240762326762
# Gradient WMA RSI Signal : Final Profit: $0.00, 0.0%

# Fitting 5 folds for each of 27 candidates, totalling 135 fits
# Best parameters: {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 500}
# Mean Squared Error after tuning: 0.14824202771314335
# Gradient WMA RSI Signal : Final Profit: $0.00, 0.0%


# %%
print(df_test['WMA RSI Signal'].value_counts())
print(df_test['Logistic WMA RSI Signal'].value_counts())
print(df_test[df_test['VPT']==0])




