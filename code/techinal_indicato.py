# %%
import numpy as np
import pandas as pd


class TechinalIndicatoHelper(): 
      

    def __calculate_wma(self,prices, window):
    # Calculate the Weighted Moving Average (WMA)

        weights = np.arange(1, window + 1)
        wma = prices.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        weights2 = np.arange(1, window*2 + 1)
        wma2 = prices.rolling(window*2).apply(lambda prices: np.dot(prices, weights2) / weights2.sum(), raw=True)
        return wma, wma2


    def __calculate_ema(self,series, span):
    # Calculate the Exponential Moving Average (EMA)
        return series.ewm(span=span, adjust=False).mean()

    def __calculate_vpt(self, df_price,df_volume, ema_period):
    # Calculate Volume Price Trend (VPT) and its EMA signal line
        #https://www.marketvolume.com/technicalanalysis/volumepricetrend.asp
        #https://www.investopedia.com/terms/v/vptindicator.asp

        #PFI(Percentage Force Index) =  Volume * (Close - ClosePrev) / ClosePrev
        PFI = (((df_price - df_price.shift(1)) / df_price.shift(1)) * df_volume)
        #VPT(Volume Price Trend) = VPT prev + PFI ,Volume Price Trend as running cumulative sum of Percentage Force index
        VPT = PFI.cumsum()
        #Exponential Moving Average to VPT as a signal line
        VPT_EMA_Signal_Line =  self.__calculate_ema(VPT, ema_period)
        return VPT, VPT_EMA_Signal_Line
    
    def __calcualte_signal_by_techinal_indicators(self, df):
    # Generate buy, sell, and neutral signals from technical indicators

        # Generate buy, sell, and neutral signals for WMA
        df['WMA Signal'] = np.where(df['Adj Close'] < df['WMA'], 1,
                                        np.where(df['Adj Close'] > df['WMA'], -1, 0))
        df['WMA2 Signal'] = np.where(df['Adj Close'] < df['WMA2'], 1,
                                        np.where(df['Adj Close'] > df['WMA2'], -1, 0))

        # Generate buy, sell, and neutral signals from VPT
        df['VPT Signal'] = 0
        df.loc[df['VPT'] > df['VPT EMA Signal Line'], 'VPT Signal'] = 1
        df.loc[df['VPT'] < df['VPT EMA Signal Line'], 'VPT Signal'] = -1

        # combine WMA and VPT signals
        df['WMA VPT Signal'] = 0
        df.loc[df['WMA Signal'] == df['VPT Signal'], 'WMA VPT Signal'] = df.loc[df['WMA Signal'] == df['VPT Signal']]['WMA Signal']

        return df
    
    def apply_techinal_indicators(self, df, window):
    #calculate techinal indicators for givan dataset, with time windows in days

        df['WMA'],df['WMA2'] = self.__calculate_wma(df['Adj Close'], window)
        df['VPT'],df['VPT EMA Signal Line'] = self.__calculate_vpt(df['Adj Close'],df['Volume'],window)

        # Calculate differences and percentage changes
        df['adj_close_diff'] = df['Adj Close'].diff()
        df['WMA_diff'] = df['WMA'].diff()
        df['VPT_diff'] = df['VPT'].diff()
        df['VPT EMA Signal Line diff'] = df['VPT EMA Signal Line'].diff()
        df['Volume diff'] = df['Volume'].diff()

        df['adj_close_pct_change'] = df['Adj Close'].pct_change() # Percentage change
        df['WMA_pct_change'] = df['WMA'].pct_change() # Percentage change
        df['VPT_pct_change'] = df['VPT'].pct_change() # Percentage change
        df['VPT_signal_pct_change'] = df['VPT EMA Signal Line'].pct_change() # Percentage change
        df['Volume_pct_change'] = df['Volume'].pct_change()

        # now base on the techinal indicators find the buy, sell and neutral signals
        df = self.__calcualte_signal_by_techinal_indicators(df)

        #clear na values
        df.dropna(inplace=True)

        return df
        # Step 8: Generate buy, sell, and neutral signals for WMA
 
    def calculate_profit(self, adj_close,signal, initial_cash=10000):
    # Simulate trades and calculate profit

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
    
    def get_dataset(self, yfh,ticker_symbol,sd,ed,window):
    # Fetch and process stock data
        stock_df = yfh.get_data(ticker_symbol,sd,ed)[['Date', 'Adj Close','Volume']]
        stock_df_with_indecators = self.apply_techinal_indicators(stock_df, window)
        stock_df_with_indecators["ticker"] = ticker_symbol
        return stock_df_with_indecators




