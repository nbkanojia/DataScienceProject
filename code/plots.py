# %%
import matplotlib.pyplot as plt
import plotly.graph_objects as go
class PlotHelper(): 
      

    def plot(self, ticker, signal, df):
   
        fig = go.Figure()

        # Plot Adjusted Close Price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

        if signal == "WMA Signal":
            fig.add_trace(go.Scatter(x=df['Date'], y=df['WMA'], mode='lines', name='WMA', line=dict(color='orange')))
        elif signal == "SMA Signal":
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA'], mode='lines', name='SMA', line=dict(color='orange')))
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

    def plot2(self, title, x,y,x_buy,y_buy,x_sell,y_sell):
    
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
            




