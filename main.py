import streamlit as st
import pandas as pd

st.title('ML Project Group 99 - StockWise')
st.subheader("Introduction")
st.write('Stock prices are known for their high levels of volatility, which can be attributed to a myriad of factors, including economic conditions, geopolitical events, investor sentiment, and company-specific news. This inherent volatility makes predicting stock prices a challenging task. Machine learning (ML) techniques have gained popularity in the realm of stock price prediction due to their ability to handle complex data patterns and adapt to changing market dynamics and uncover hidden patterns and relationships that human analysts may overlook. The benefits of using ML for stock price prediction include improved accuracy compared to traditional methods, the ability to process vast amounts of data in real time, and the potential for automation, enabling investors to make more informed decisions and better manage risk in the highly dynamic world of finance.')
st.subheader("Literature Review")
st.write("Most algorithms used in stock price prediction are some forms of time series analysis models, recurrent neural networks, or deep learning models. Out of many time series models, Autoregressive Integrated Moving Average is very popular for modeling and forecasting time series data, including stock prices. They can capture trends, seasonality, and autocorrelation in the data and are stronger at producing short-term predictions [2]. The Long-Short Term Memory network is an advanced recurrent neural network that is stronger for long-term stock prediction due to its ability to remember sequential information and have “contextual memory” for past information [1].")
st.subheader("Problem Statement")
st.write("We aim to predict the future price movements of stocks relative to the future price movement of a synthetic index composed of NASDAQ-listed stocks. Specifically, we are looking to accurately predict the weighted average price 1 minute into the future.")
st.subheader("Dataset")
st.write("We chose the Kaggle dataset from the competition “Optiver - Trading at the Close: Predict US stocks closing movements” which contains historical data for the daily ten-minute closing auction on the NASDAQ stock exchange. The training dataset has 5,237,981 rows of various stocks and their features and the target price after 1 minute. The test dataset has 33000 test cases. There are 15 features in the dataset, some of which include the stock_id, date_id, matched_size which is the amount that can be matched at the current reference price (in USD), weight average price, and other features.")
st.subheader("Methods")
st.write("First, we will develop a baseline model by running a deep neural network on all features without any feature extraction or dimension reduction.")
st.write("We will then try to extract features using PCA and then train an LSTM and then an ARIMA on these extracted features.")
st.subheader("Results & Discussion")
st.write("To quantitatively assess the performance of our machine learning models, we have selected a set of metrics that collectively provide a robust framework for assessing the effectiveness and reliability of our research outcomes. We will use mean absolute error (MAE) to provide a sense of the magnitude of errors in predictions. Mean squared error (MSE) will be employed since it penalizes larger errors more heavily than MAE. We will also use r-squared (R2) since it indicates goodness of fit.")

st.subheader("Gantt Chart")
url = "https://gtvault-my.sharepoint.com/:x:/g/personal/ecai32_gatech_edu/EcyEwpm5R8lNgnS7mIxyW1YBPNsGfQDo-N4J-dkUq9hfmg?e=4%3Auiq637&fromShare=true&at=9&wdLOR=cF89E960B-265C-544C-BE4F-AAA752B8557D"
st.write("Here is our [Gantt Chart](%s)" % url)
