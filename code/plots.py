# %%
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
class PlotHelper(): 
      

    def plot(self, ticker, signal, df):
   
        fig = go.Figure()

        # Plot Adjusted Close Price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

        if signal == "WMA Signal":
            fig.add_trace(go.Scatter(x=df['Date'], y=df['WMA'], mode='lines', name='WMA', line=dict(color='orange')))
      
        df_buy_signals = df[df[signal] == 1]
        df_sell_signals = df[df[signal] == -1]
        
        fig.add_trace(go.Scatter(x=df_buy_signals['Date'], y=df_buy_signals['Adj Close'], mode='markers', name='Buy Signal',
                                marker=dict(symbol='triangle-up', size=7, color='green')))
        fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['Adj Close'], mode='markers', name='Sell Signal',
                                marker=dict(symbol='triangle-down', size=7, color='red')))
        
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
    

    def plot_vpt(self,ticker, df):
        fig = go.Figure()

        # Plot Adjusted Close Price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

        df_buy_signals = df[df["VPT Signal"] == 1]
        df_sell_signals = df[df["VPT Signal"] == -1]
        
        fig.add_trace(go.Scatter(x=df_buy_signals['Date'], y=df_buy_signals['Adj Close'], mode='markers', name='Buy Signal',
                                marker=dict(symbol='triangle-up', size=7, color='green')))
        fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['Adj Close'], mode='markers', name='Sell Signal',
                                marker=dict(symbol='triangle-down', size=7, color='red')))
    
        
        #fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['VPT'], name='VPT',mode='lines', line=dict(color='blue')))
        #fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['VPT EMA Signal Line'],name='Signal Line', mode='lines', line=dict(color='blue')))
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis {"VPT Signal"}',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            xaxis=dict(rangeslider=dict(visible=True)),
            template='plotly_dark',
            height=600,
            width=1000
        )

        fig.show()

        fig = go.Figure()

       

        df_buy_signals = df[df["VPT Signal"] == 1]
        df_sell_signals = df[df["VPT Signal"] == -1]
        
        
            
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VPT'], name='VPT',mode='lines', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VPT EMA Signal Line'],name='Signal Line', mode='lines', line=dict(color='red')))
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis {"VPT Signal"}',
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

    def plot_correlation_matrix(self, df):
        # plot correlation matrix as heatmap
        plt.figure(figsize=(10,10))
        sns.set_style("whitegrid")
        df_cluster2 = df.corr(numeric_only=True)
        sns.heatmap(df_cluster2,
                    cmap='RdYlBu',
                    annot=True,
                    linewidths=0.2,
                    linecolor='lightgrey').set_facecolor('white')
        plt.title("Correlation Analysis")
        
    def plot_distribution(self,x_name,df):
        plt.figure(figsize=(10, 4))
        sns.countplot(x=x_name, data=df, hue=x_name)
        plt.title("Distribution of "+ x_name)
        plt.xlabel(x_name)
        plt.ylabel("Frequency")
        plt.show()
            




