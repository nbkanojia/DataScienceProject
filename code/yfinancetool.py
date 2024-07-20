# %%
import yfinance as yf
import os.path
import pandas as pd
class YFinanceHelper(): 
      
    def __init__(self, basepath): 
        self.basepath = basepath 
    def get_data(self, ticker,start_date,end_date):
        fname = self.basepath +ticker+"_"+ start_date+ "_" + end_date +".csv"
        if not os.path.isfile(fname):
            df = yf.download(ticker ,start=start_date, end=end_date)
            df.to_csv(fname)
            
        df = pd.read_csv(fname)
        df['Date'] = pd.to_datetime(df['Date'])
        return df 




