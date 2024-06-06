# DataScienceProject

# Stock Market Analysis Project

## Overview
This project aims to analyze historical stock market data for a selected company. The objective is to develop machine learning models to predict stock prices and assess trends in the market.

## Dataset Description
The dataset contains daily historical stock prices for the selected company. Each row represents a trading day's data, with columns for the date, opening price, closing price, highest price, lowest price, adjusted closing price, and trading volume. The data spans approximately ten years and includes around 2,500 records.

## Directory Structure
The project directory is organized as follows:
-root<br />
|-- data<br />
|   |-- AAPL_2010-2020_daily_data.csv<br />
|   |-- TSLA_2010-2020_daily_data.csv<br />
|   |-- BABA_2010-2020_daily_data.csv<br />
|-- code<br />
|   |-- yfinancetool.py<br />
|   |-- stockmarketML.py<br />
|-- README.md<br />

<b>.csv files:</b> These will be named according to the stock ticker symbol and the data range, for example: AAPL_2010-2020_daily_data.csv. <br />
<b>yfinancetool.py:</b> This file contains the script for downloading data from Yahoo Finance using the yfinance library. <br />
<b>stockmarketML.py:</b> This file contains the machine learning code for training and testing different models on the stock market data. <br />
<b>readme.md:</b> This file provides information about the project, including how to use the code and data, and any other relevant details. <br />

## Data Source and Collection
The data is collected from Yahoo Finance using the yfinance Python library. It is publicly available and intended for research and educational purposes. The yfinance library is distributed under the Apache Software License. Users should refer to Yahoo's terms of use for details on their rights to use the actual data downloaded.

## Metadata
The dataset includes columns for the date, opening price, closing price, highest price, lowest price, adjusted closing price, and trading volume.
