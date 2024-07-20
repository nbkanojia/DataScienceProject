# %%
import numpy as np
import pandas as pd

class TechinalIndicatoHelper(): 
      

    def __calculate_wma(self,prices, window):
        weights = np.arange(1, window + 1)
        wma = prices.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        return wma

    def __calculate_sma(self, prices, window):
        sma = prices.rolling(window=window).mean()
        return sma
    # Function to calculate Relative Strength Index (RSI)
    def __calculate_rsi(self,prices, window):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def __calculate_bollinger_bands(self,prices, window, num_of_std=2):
        SMA= prices.rolling(window=window).mean()
        STD = prices.rolling(window=window).std()
        UpperBand = SMA + (STD * num_of_std)
        LowerBand = SMA - (STD * num_of_std)
        return UpperBand,LowerBand

    def __calculate_percentage_change(self,prices, window):

        pc = []
        
        for i in range(len(prices)):
            if i < window:
                pc.append(pd.NA)
            else:
                start_price = prices[i - window]
                end_price = prices[i]
                pc.append((end_price - start_price) / start_price * 100)
        return pc

    def __calculate_ema(self,series, span):
        return series.ewm(span=span, adjust=False).mean()

    def __calculate_vpt(self,df, ema_period=20):
        df['PFI'] = df['Volume'] * (df['Adj Close'].diff() / df['Adj Close'].shift(1))
        df['VPT'] = df['PFI'].cumsum()
        df['VPT_EMA'] = self.__calculate_ema(df['VPT'], ema_period)
        return df
    
    def __calcualte_signal_by_techinal_indicators(self, df):
        # df['WMA Signal'] = np.where(df['Adj Close'] < df['WMA'], 1,
        #                                 np.where(df['Adj Close'] > df['WMA'], -1, 0))
        df['WMA Signal'] = np.where(df['Adj Close'] < df['WMA'], 1,
                                        np.where(df['Adj Close'] > df['WMA'], -1, 0))
        
        df['SMA Signal'] = np.where(df['Adj Close'] < df['SMA'], 1,
                                        np.where(df['Adj Close'] > df['SMA'], -1, 0))
        
        # Step 9: Generate buy, sell, and neutral signals for RSI
        df['RSI Signal'] = np.where(df['RSI'] < 30, 1,
                                        np.where(df['RSI'] > 70, -1, 0))

        
        # Step 9: Generate buy, sell, and neutral signals for RSI

        df['PC Signal'] = np.where(df['PC'] <= -8, 1,
                                        np.where(df['PC'] >= 8, -1, 0))#-5-15


        df['BB Signal'] = np.where(df['Adj Close'] < df['BLB'], 1,
                                        np.where(df['Adj Close'] > df['BUB'], -1, 0))

        df['VPT Signal'] = 0
        df.loc[df['VPT'] > df['VPT_EMA'], 'VPT Signal'] = -1
        df.loc[df['VPT'] < df['VPT_EMA'], 'VPT Signal'] = 1

        # df['WMA RSI Signal'] = np.where((df['WMA Signal'] == 1) & (df['RSI Signal']  == 1) & (df['VPT Signal']  == 1), 1,
        #                                 np.where((df['WMA Signal'] == -1) & (df['RSI Signal']  == -1) & (df['VPT Signal']  == -1), -1, 0))
    
        df['WMA RSI Signal'] = np.where((df['WMA Signal'] == 1) & (df['VPT Signal']  == 1), 1,
                                        np.where((df['WMA Signal'] == -1) & (df['VPT Signal']  == -1), -1, 0))
    
#new_df['All Signal'] = np.where((new_df['WMA Signal'] == 'Buy') & (new_df['RSI Signal'] == 'Buy') & (new_df['Fib Signal'] == 'Buy'), 'Buy',
#                                np.where((new_df['WMA Signal'] == 'Sell') & (new_df['RSI Signal'] == 'Sell') & (new_df['Fib Signal'] == 'Sell'), 'Sell', 'Neutral'))

#new_df['All Signal'] = np.where((new_df['WMA Signal'] == 'Buy') & (new_df['RSI Signal'] == 'Buy') & (new_df['Fib Signal'] == 'Buy'), 'Buy',
#                                np.where((new_df['WMA Signal'] == 'Sell') & (new_df['RSI Signal'] == 'Sell') & (new_df['Fib Signal'] == 'Sell'), 'Sell', 'Neutral'))
        
        return df
    def apply_techinal_indicators(self, df, window):
        df['SMA'] = self.__calculate_sma(df['Adj Close'], window)
        df['WMA'] = self.__calculate_wma(df['Adj Close'], window)

        df['RSI'] = self.__calculate_rsi(df['Adj Close'], window)

        df['BUB'],df['BLB'] = self.__calculate_bollinger_bands(df['Adj Close'], window)

        df['PC'] = self.__calculate_percentage_change(df['Adj Close'], window)

        df['VPT'] = ((df['Adj Close'].diff() / df['Adj Close'].shift(1)) * df['Volume']).cumsum()
        #df['PFI'] = df['Volume'] * (df['Adj Close'].diff() / df['Adj Close'].shift(1))
        #df['VPT'] = df['PFI'].cumsum()
        df['VPT_EMA'] = self.__calculate_ema(df['VPT'], window)

        df = self.__calcualte_signal_by_techinal_indicators(df)
        return df
        # Step 8: Generate buy, sell, and neutral signals for WMA
 
    def calculate_profit(self, adj_close,signal, initial_cash=10000):
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




