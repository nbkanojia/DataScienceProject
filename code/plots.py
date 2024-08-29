# %%
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
class PlotHelper(): 
      
    def __init__(self): 
        # Set font scale for plots
        sns.set(font_scale=2)
        # Set default DPI for all plots
        self.dpi = 100

    def plot(self, ticker, signal, df):
    # Plot stock price with buy/sell signals

        fig = go.Figure()

        # Plot Adjusted Close Price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

        # Plot WMA if the signal is WMA
        if signal == "WMA Signal":
            fig.add_trace(go.Scatter(x=df['Date'], y=df['WMA'], mode='lines', name='WMA', line=dict(color='orange')))
      
        # Filter data for buy and sell signals
        df_buy_signals = df[df[signal] == 1]
        df_sell_signals = df[df[signal] == -1]
        
        # Plot buy signals as markers
        fig.add_trace(go.Scatter(x=df_buy_signals['Date'], y=df_buy_signals['Adj Close'], mode='markers', name='Buy Signal',
                                marker=dict(symbol='triangle-up', size=7, color='green')))
        # Plot Sell signals as markers
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
    # Plot VPT (Volume Price Trend) with buy/sell signals

        fig = go.Figure()

        # Plot Adjusted Close Price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))

        # Filter data for buy and sell signals
        df_buy_signals = df[df["VPT Signal"] == 1]
        df_sell_signals = df[df["VPT Signal"] == -1]
        
        # Plot buy signals as markers
        fig.add_trace(go.Scatter(x=df_buy_signals['Date'], y=df_buy_signals['Adj Close'], mode='markers', name='Buy Signal',
                                marker=dict(symbol='triangle-up', size=7, color='green')))
        
        # Plot Sell signals as markers
        fig.add_trace(go.Scatter(x=df_sell_signals['Date'], y=df_sell_signals['Adj Close'], mode='markers', name='Sell Signal',
                                marker=dict(symbol='triangle-down', size=7, color='red')))
    
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

        # Plot VPT and Signal Line    
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
        

    def plot_correlation_matrix(self, df):
    # Plot correlation matrix
        
        sns.set(font_scale=1.2)
        plt.figure(figsize=(10,10), dpi=self.dpi)
        sns.set_style("whitegrid")
        df_cluster2 = df.corr(numeric_only=True).round(2)
        # plot correlation matrix as heatmap
        sns.heatmap(df_cluster2,
                    cmap='RdYlBu',
                    annot=True,
                    linewidths=0.2,
                    linecolor='lightgrey').set_facecolor('white')
        plt.title("Correlation Analysis")
        plt.show()
        sns.set(font_scale=2)

    def plot_confusion_matrix(self, y_test, y_pred):
    # Plot confusion matrix for model predictions
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
       
        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(8, 6), dpi=self.dpi)
       
        labels = ["Sell(-1)", "Neutral(0)", "Buy(1)"] 
        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Confusion Matrix")
        plt.show()
        
        
    def plot_distribution(self,x_name,df):
    # Plot the distribution of a specific column

        plt.figure(figsize=(10, 4), dpi=self.dpi)
        sns.countplot(x=x_name, data=df, hue=x_name)
        plt.title("Distribution of "+ x_name)
        plt.xlabel(x_name)
        plt.ylabel("Frequency")
        plt.show()

    def plot_feature_selection_score(self,columns,scores):
    # Plot feature selection scores
        df_feature = pd.DataFrame()
        df_feature["columns"] = columns
        df_feature["scores"]= scores
        df_feature= df_feature.sort_values(by=['scores'],ascending=False)
        plt.figure(figsize=(8, 6), dpi=self.dpi)
        sns.barplot(y=df_feature["columns"], x=df_feature["scores"])
        plt.xlabel('F-Score')
        plt.ylabel('Features')
        plt.title("F-Scores for Features using f_classif")
        plt.show()
        
    def plot_class_distribution(self,stock_df):
    # Plot class distribution as a pie chart
        plt.figure(figsize=(5,5),dpi=self.dpi)
        plt.pie(stock_df['WMA VPT Signal'].value_counts(), labels=['Neutral(0)', 'Sell(-1)', 'Buy(1)'], autopct='%.0f%%') 
        plt.title("Class distribution")
        plt.show()
    
    def plot_profit_compare(self, title, profit_curr, profit_pct, hue_model ):
    # Plot comparison of profit for different models
        plt.figure(figsize=(4, 6), dpi=100)
        ax1= sns.barplot(y=profit_curr, hue=hue_model, legend=True)
        for i in ax1.containers:
            ax1.bar_label(i,labels=(profit_curr.apply(str) + " (" + profit_pct.apply(str)+"%)"))
        ax1.set_ylabel("$")
        ax1.set_xlabel("")
        ax1.set_title(title, y=1.05)
        plt.show()
    
    def plot_profit_compare3(self, title, df ):
    # Plot comparison of profit with bar labels
         
        plt.figure(figsize=(8, 6))
        barplot = sns.barplot(x='model', y='count', hue='model', data=df, legend=True)

        # Add the string labels on top of the bars
        for index, row in df.iterrows():
            barplot.text(index, row['count'] + 10, row['str_val'], color='black', ha="center")

        # Set the labels and title
        plt.xlabel('')
        plt.ylabel('Profit ($)')
        plt.title(title,y=1.06)

        # Display the plot
        plt.show()

    def plot_signals(self,title, df ):
    # Plot signals with a barplot

        sns.set(font_scale=1.8)
        plt.figure(figsize=(6, 5), dpi=100)
        ax = sns.barplot(df, x="tread", y="count", hue="model", legend=True)
        for i in ax.containers:
            ax.bar_label(i,)
        plt.title(title, y=1.05)

        plt.show()
        sns.set(font_scale=2)

    def plot_profit_trend(self,title,x,y):
    # Plot profit trend over time

        fig = plt.figure(figsize=(6, 4), dpi=100)
        sns.lineplot(y=y,x=x)
        plt.ylabel("profit($)")
        plt.xlabel("Date")
        plt.title(title)
        fig.autofmt_xdate(rotation=45)
        plt.show()


