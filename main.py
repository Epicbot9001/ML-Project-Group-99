import streamlit as st
import pandas as pd

st.title('ML Project Group 99 - StockWise')

st.subheader("Introduction")
st.write('Stock prices are known for their high levels of volatility, which can be attributed to a myriad of factors, including economic conditions, geopolitical events, investor sentiment, and company-specific news. This inherent volatility makes predicting stock prices a challenging task. Machine learning (ML) techniques have gained popularity in the realm of stock price prediction due to their ability to handle complex data patterns and adapt to changing market dynamics and uncover hidden patterns and relationships that human analysts may overlook. The benefits of using ML for stock price prediction include improved accuracy compared to traditional methods, the ability to process vast amounts of data in real-time, and the potential for automation, enabling investors to make more informed decisions and better manage risk in the highly dynamic world of finance. Most algorithms used in stock price prediction are some forms of time series analysis models, recurrent neural networks, or deep learning models. Out of many time series models, Autoregressive Integrated Moving Average is very popular for modeling and forecasting time series data, including stock prices. They can capture trends, seasonality, and autocorrelation in the data and are stronger at producing short-term predictions [2]. The Long-Short Term Memory network is an advanced recurrent neural network that is stronger for long-term stock prediction due to its ability to remember sequential information and have “contextual memory” for past information [1].')
kaggle_url = "https://www.kaggle.com/competitions/optiver-trading-at-the-close/data"
dataset_url = "https://gtvault-my.sharepoint.com/:u:/g/personal/dliu382_gatech_edu/EfwO1X9XhmRGhYZiwRfP7doBikNXZLwGvQHattgw6LvazQ?e=BZ2v1w"
st.write('Click [here](%s) for where we found the dataset' % kaggle_url)
st.write('Click [here](%s) for accessing our dataset' % dataset_url)

st.subheader("Problem Definition")
st.write("With the challenging nature of predicting the future price movements of stocks, we aim to predict these movements relative to the future price movement of a synthetic index composed of NASDAQ-listed stocks. Specifically, we are looking to accurately predict the weighted average price 1 minute into the future.")

st.subheader("Methods")
st.markdown(f"**Preprocessing:**")
st.write("""
- To clean the dataset we are working with, we removed rows that contained null and nand values as well as outliers. We chose to remove rows instead of replacing them with the mean or 0 because our dataset was extremely large, so we thought that removing a couple of rows would not hurt as the Kaggle dataset was mostly clean. Then, we checked if the datatypes were consistent and normalized the data to be between 0 and 1 which helps the neural network train. With this, we were able to reduce the size of the large dataset and ease the training and optimization processes.
""")
st.markdown(f"**Model:**")
st.write("""
- We initially developed a baseline model by running a deep neural network on all features without any feature extraction or dimension reduction. This means that we used the preprocessed data mentioned above. A sequential model is used which is a linear stack of layers. The first layer is a Dense layer with 512 neurons that uses ReLu activation and L2 regularization with a coefficient of 0.001. Following this layer, we have three more Dense layers each with decreasing units of 128, 32, and 8 respectively all of which use ReLu activation and L2 regularization to further prevent overfitting. The model’s output layer is a single neuron with a linear activation function since this is a regression task that requires a singular output and a linear activation function. All of which is suitable for regression tasks that predict a continuous value which further validates the usage of this model for a task like stock price predictions. We initially set the learning rate for the optimizer at 0.0005 and the loss function used Mean Squared Error (MSE). This model architecture is quite simple and a good starting point for this task.
""")

st.subheader("Results/Discussion")
st.write("Here are the graphs of the loss and metrics that we evaluated our neural network on.")
st.image("training_metrics.png")
st.image("losses.png")

st.write("""
We use 3 metrics to evaluate our model:
1. Mean Squared Error: We use this as our loss function since it penalizes larger errors more heavily than MAE.
2. Mean Absolute Error: This is the metric that the Kaggle leaderboard (from where we picked our dataset) uses to evaluate the model. The lower it is, the better.
3. Custom Accuracy Function: We made a custom function to measure our model’s accuracy. We measure the accuracy based on the percent difference (absolute) between the predicted and actual values. The higher it is, the better.
""")
         
st.write("There are many potential reasons why our model has a MAE of 5.433. As we decided to use a baseline model with no dimensionality reduction, the model has to train on every feature. This means that the improvements that are made between each epoch will be small thus the model itself will take a long time and also place equal weights on certain features that are highly correlated with each other. Our model would have also performed better if we employed feature selection so that our model would only be training and learning from relevant features. A neural network may also not be optimal for stock price prediction because of a lack of causality. Neural networks are primarily designed for pattern recognition, but in the financial markets, correlations don't always imply causation. Predicting stock prices based on historical data alone might not capture underlying economic and business factors. Neural networks may perform better at short-term price predictions (minutes or hours) but struggle with long-term predictions (days, weeks, or months) due to the increased influence of external factors. We hope to tackle the issue of long-term predictions through the LSTM since it has a component that specializes in capturing long-term context.")
st.write("Additionally, stock price prediction is a challenging topic because stocks fluctuate so much, making them hard to predict. Oftentimes, stock price prediction is a coin flip of guessing whether it will go up or down in price. When setting the threshold of our custom metric function to 1, basically evaluating the model on whether we predicted the stock price movement direction correctly, our model achieved a 52% accuracy which means that we are able to guess whether the stock price goes up or down 2% better than a coin flip which is impressive. The current leaderboard in the Kaggle competition reflects similar results to our neural network with the top model having a MAE of 5.3173. Given all of this, we believe that our current neural network provides an excellent benchmark for which to evaluate future models that we plan on training and examining.")

st.subheader("Reflections/Next Steps")
st.write("In terms of the next steps, we hope to modify our training process to pinpoint the cause of the high loss. For instance, we can continue adjusting the learning rate, the regularization constant, and the number of nodes and layers in the neural network. Additionally, we could revisit the activation functions and batch sizes and ultimately ensure our data normalization is effectively applied. Another option is testing different optimizers and incorporating regularization techniques. To prevent overfitting, we could incorporate early stopping as we train our model. Additionally, we will implement the LSTM and ARIMA models according to our plans to determine if they can perform better at stock price prediction because of their specializations in long-term contextual capabilities and short-term price prediction respectively.")

st.subheader("Gantt Chart")
gantt_url = "https://gtvault-my.sharepoint.com/:x:/g/personal/ecai32_gatech_edu/EcyEwpm5R8lNgnS7mIxyW1YBPNsGfQDo-N4J-dkUq9hfmg?e=4%3Auiq637&fromShare=true&at=9&wdLOR=cF89E960B-265C-544C-BE4F-AAA752B8557D"
st.write("Here is our [Gantt Chart](%s)" % gantt_url)

st.subheader("Contribution Chart")
st.image("contribution.png")
contribution_url = "https://docs.google.com/spreadsheets/d/1Rgvp4U7Ef4F_N1EJv0jqYG-QiYhiDy6liCCfzj7GoTY/edit#gid=0"
st.write("Click [here](%s) to access our Contribution Chart" % contribution_url)


