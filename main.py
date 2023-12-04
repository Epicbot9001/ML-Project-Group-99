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
st.markdown(f"**Model 1 (Neural Network):**")
st.write("""
- We initially developed a baseline model by running a deep neural network on all features without any feature extraction or dimension reduction. This means that we used the preprocessed data mentioned above. A sequential model is used which is a linear stack of layers. The first layer is a Dense layer with 512 neurons that uses ReLu activation and L2 regularization with a coefficient of 0.001. Following this layer, we have three more Dense layers each with decreasing units of 128, 32, and 8 respectively all of which use ReLu activation and L2 regularization to further prevent overfitting. The model’s output layer is a single neuron with a linear activation function since this is a regression task that requires a singular output and a linear activation function. All of which is suitable for regression tasks that predict a continuous value which further validates the usage of this model for a task like stock price predictions. We initially set the learning rate for the optimizer at 0.0005 and the loss function used Mean Squared Error (MSE). This model architecture is quite simple and a good starting point for this task.
- """ f"**Results/Discussion**" """
    - Here are the graphs of the loss and metrics that we evaluated our neural network on.
""")

# st.subheader("Results/Discussion")
# st.write("Here are the graphs of the loss and metrics that we evaluated our neural network on.")
st.image("training_metrics_nn.png")
st.image("losses_nn.png")

st.write("""
- We use 3 metrics to evaluate our model:
    1. Mean Squared Error: We use this as our loss function since it penalizes larger errors more heavily than MAE.
    2. Mean Absolute Error: This is the metric that the Kaggle leaderboard (from where we picked our dataset) uses to evaluate the model. The lower it is, the better.
    3. Custom Accuracy Function: We made a custom function to measure our model’s accuracy. We measure the accuracy based on the percent difference (absolute) between the predicted and actual values. The higher it is, the better.
- There are many potential reasons why our model has a MAE of 5.433. As we decided to use a baseline model with no dimensionality reduction, the model has to train on every feature. This means that the improvements that are made between each epoch will be small thus the model itself will take a long time and also place equal weights on certain features that are highly correlated with each other. Our model would have also performed better if we employed feature selection so that our model would only be training and learning from relevant features. A neural network may also not be optimal for stock price prediction because of a lack of causality. Neural networks are primarily designed for pattern recognition, but in the financial markets, correlations don't always imply causation. Predicting stock prices based on historical data alone might not capture underlying economic and business factors. Neural networks may perform better at short-term price predictions (minutes or hours) but struggle with long-term predictions (days, weeks, or months) due to the increased influence of external factors. We hope to tackle the issue of long-term predictions through the LSTM since it has a component that specializes in capturing long-term context.
- Additionally, stock price prediction is a challenging topic because stocks fluctuate so much, making them hard to predict. Oftentimes, stock price prediction is a coin flip of guessing whether it will go up or down in price. When setting the threshold of our custom metric function to 1, basically evaluating the model on whether we predicted the stock price movement direction correctly, our model achieved a 52% accuracy which means that we are able to guess whether the stock price goes up or down 2% better than a coin flip which is impressive. The current leaderboard in the Kaggle competition reflects similar results to our neural network with the top model having a MAE of 5.3173. Given all of this, we believe that our current neural network provides an excellent benchmark for which to evaluate future models that we plan on training and examining.
""")
         
st.markdown(f"**Model 2 (LSTM):**")
st.write("""
- We developed a sequential model with an LSTM layer of 128 units and a dense neural network layer of 1 node for the output. In addition to using the preprocessed data from above, we also had to manipulate the data into time sequences since LSTMs require time sequence data. This means we needed to change our 2D data of row x features to 3D. We accomplished this by grouping the data into each stock and creating a new dimension for each different type of stock that contained row x features where each row is the stock fluctuation at the given timestep. We used a ReLu activation function for the LSTM layer and an L2 regularization rate of 0.001. For the dense layer, we used a linear activation function since this is a regression task that requires a singular output. We initially set the learning rate for the optimizer at 0.0005 and the loss function used Mean Squared Error (MSE). We believe that the LSTM should perform better because of its ability to process sequential data, which could help track temporal patterns in stock prices.
- """ f"**Results/Discussion**" """
    - Here are the graphs of the loss and metrics that we evaluated our neural network on.
""")

st.image("training_metrics_lstm.png")
st.write("We use the same 3 metrics to evaluate our model: Mean Squared Error, Mean Absolute Error, Custom Accuracy Function.")
st.image("losses_lstm.png")

st.write("""
- Our model had a final MAE of 5.376 which was an improvement from the neural network baseline model. For reference again, the current leaderboard in the Kaggle competition reflects similar results to our neural network with the top model having an MAE of 5.3173. The LSTM model's performance on stock price prediction shows an overall improvement over epochs. The validation loss decreases from 66.78788 to 50.99999, indicating that the model is better at generalizing to unseen data. The validation loss decreases rapidly in the early epochs but then plateaus towards the end of training. This suggests that the model may have reached a local optimum and further training may not improve its performance significantly.
- The mean absolute error (MAE) also decreases from 7.1234 to 5.37666, suggesting that the model's predictions are more accurate. The MAE decreases more slowly than the validation loss. This suggests that the model is still struggling to make very accurate predictions, even though it is generalizing better to unseen data. The custom metric, which measures the model's ability to capture long-term trends, increases from 4.156 to 8.333, indicating that the model is better at identifying long-term patterns in the data. Overall, these results suggest that the LSTM model is a promising tool for stock price prediction. However, further work is needed to improve the model's performance and make it more robust to market fluctuations. Predicting stock prices based on historical data alone might not capture underlying economic and business factors. If provided with other world event data, the LSTM may prove to be a stronger model.
""")
         
st.subheader("Comparison")
st.write("Comparing the two models, it seems that the LSTM had a slight advantage over the traditional neural network. This could be attributed to the fact that the LSTM has a memory component and can retrieve information stored from the past and handles temporal data well which aligns with stock price data. We conjecture that the LSTM did not outperform the neural network by a significant margin because the stock price data may not have had a significant correlation between the time and the stock price. Hence, an LSTM would not have an accurate prediction even with the ability to understand temporal trends in data. Additionally, better feature engineering may be able to boost scores for both the LSTM and neural network since feature engineering is imperative for stock price prediction in order to establish clear correlations between features and stock price so that the neural network or LSTM can model.")

st.subheader("Reflections/Next Steps")
st.write("In terms of the next steps for the neural network, we hope to modify our training process to pinpoint the cause of the high loss. For instance, we can continue adjusting the learning rate, the regularization constant, and the number of nodes and layers in the neural network. Additionally, we could revisit the activation functions and batch sizes and ultimately ensure our data normalization is effectively applied. Another option is testing different optimizers and incorporating regularization techniques. To prevent overfitting, we could incorporate early stopping as we train our model.")
st.write("For the LSTM, we want to experiment with training LSTMs for each individual stock since we believe that each individual stock may behave differently, and when we train an LSTM on all the stocks, the LSTM has trouble generalizing predictions for all stocks. We also found that LSTM training time took significantly longer than training neural networks. Hence, if we could gain access to GPUs to boost the training speed, we would be able to experiment with training larger sequential models with multiple LSTM layers with even more nodes. Having a more complex model may be able to score higher on the metrics.")

st.subheader("Gantt Chart")
gantt_url = "https://gtvault-my.sharepoint.com/:x:/g/personal/ecai32_gatech_edu/EcyEwpm5R8lNgnS7mIxyW1YBPNsGfQDo-N4J-dkUq9hfmg?e=4%3Auiq637&fromShare=true&at=9&wdLOR=cF89E960B-265C-544C-BE4F-AAA752B8557D"
st.write("Here is our [Gantt Chart](%s)" % gantt_url)

st.subheader("Contribution Chart")
st.image("contribution.png")
contribution_url = "https://docs.google.com/spreadsheets/d/1Rgvp4U7Ef4F_N1EJv0jqYG-QiYhiDy6liCCfzj7GoTY/edit#gid=0"
st.write("Click [here](%s) to access our Contribution Chart" % contribution_url)

st.subheader("References")
st.write("""
LSTM/RNN/CNN Sliding Window
- https://ieeexplore.ieee.org/abstract/document/8126078/ 
[1] S. Selvin, R. Vinayakumar, E. A. Gopalakrishnan, V. K. Menon, and K. P. Soman, “Stock price prediction using LSTM, RNN and CNN-sliding window model,” 2017 International Conference on Advances in Computing, Communications and Informatics (ICACCI), Sep. 2017, doi: https://doi.org/10.1109/icacci.2017.8126078.  

ARIMA
- https://ieeexplore.ieee.org/abstract/document/7046047/ 
[2] A. A. Ariyo, A. O. Adewumi, and C. K. Ayo, “Stock Price Prediction Using the ARIMA Model,” 2014 UKSim-AMSS 16th International Conference on Computer Modelling and Simulation, Mar. 2014, doi: https://doi.org/10.1109/uksim.2014.67.

Deep Learning
- https://durham-repository.worktribe.com/output/1389193 
[3] E. Chong, C. Han, and F. C. Park, “Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies,” Expert Systems with Applications, vol. 83, pp. 187–205, Oct. 2017, doi: https://doi.org/10.1016/j.eswa.2017.04.030.

LSTM + feature selection
- https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00333-6 
[4] J. Shen and M. O. Shafiq, “Short-term stock market price trend prediction using a comprehensive deep learning system,” Journal of Big Data, vol. 7, no. 1, Aug. 2020, doi: https://doi.org/10.1186/s40537-020-00333-6.

SVM + hybrid feature selection
- https://www.sciencedirect.com/science/article/pii/S0957417409001560?via%3Dihub 
[5] M.-C. Lee, “Using support vector machine with a hybrid feature selection method to the stock trend prediction,” Expert Systems with Applications, vol. 36, no. 8, pp. 10896–10904, Oct. 2009, doi: https://doi.org/10.1016/j.eswa.2009.
02.038.
""")

