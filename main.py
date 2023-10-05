import streamlit as st
import pandas as pd

# ------ PART 1 ------

df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)
st.title('ML Project Group 99 - StockWise')
st.subheader("Introduction")
st.text('Stock prices are known for their high levels of volatility, which can be attributed to a myriad of factors, including economic conditions, geopolitical events, investor sentiment, and company-specific news. This inherent volatility makes predicting stock prices a challenging task. Machine learning (ML) techniques have gained popularity in the realm of stock price prediction due to their ability to handle complex data patterns and adapt to changing market dynamics and uncover hidden patterns and relationships that human analysts may overlook. The benefits of using ML for stock price prediction include improved accuracy compared to traditional methods, the ability to process vast amounts of data in real time, and the potential for automation, enabling investors to make more informed decisions and better manage risk in the highly dynamic world of finance.')
st.subheader("Literature Review")
st.text("Most algorithms used in stock price prediction are some forms of time series analysis models, recurrent neural networks, or deep learning models. Out of many time series models, Autoregressive Integrated Moving Average is very popular for modeling and forecasting time series data, including stock prices. They can capture trends, seasonality, and autocorrelation in the data and are stronger at producing short-term predictions [2]. The Long-Short Term Memory network is an advanced recurrent neural network that is stronger for long-term stock prediction due to its ability to remember sequential information and have “contextual memory” for past information [1].")
st.subheader("Problem Statement")
st.text("We aim to predict the future price movements of stocks relative to the future price movement of a synthetic index composed of NASDAQ-listed stocks. Specifically, we are looking to accurately predict the weighted average price 1 minute into the future.")
st.subheader("Dataset")
st.text("We chose the Kaggle dataset from the competition “Optiver - Trading at the Close: Predict US stocks closing movements” which contains historical data for the daily ten-minute closing auction on the NASDAQ stock exchange. The training dataset has 5,237,981 rows of various stocks and their features and the target price after 1 minute. The test dataset has 33000 test cases. There are 15 features in the dataset, some of which include the stock_id, date_id, matched_size which is the amount that can be matched at the current reference price (in USD), weight average price, and other features.")
st.subheader("Methods")
st.text("First, we will develop a baseline model by running a deep neural network on all features without any feature extraction or dimension reduction.")
st.text("We will then try to extract features using PCA and then train an LSTM and then an ARIMA on these extracted features.")
st.subheader("Results & Discussion")
st.text("To quantitatively assess the performance of our machine learning models, we have selected a set of metrics that collectively provide a robust framework for assessing the effectiveness and reliability of our research outcomes. We will use mean absolute error (MAE) to provide a sense of the magnitude of errors in predictions. Mean squared error (MSE) will be employed since it penalizes larger errors more heavily than MAE. We will also use r-squared (R2) since it indicates goodness of fit.")

# Display text
st.text('Fixed width text')
st.markdown('_**Markdown**_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True

# Interactive widgets
st.button('Hit me')
st.data_editor(df)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')

# -- add download button (start) --
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)
# -- add download button (end) --

st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')

# ------ PART 2 ------

data = pd.read_csv("employees.csv")

# Display Data
st.dataframe(data)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric('My metric', 42, 2)

# Media
st.image('./smile.png')

# Display Charts
st.area_chart(data[:10])
st.bar_chart(data[:10])
st.line_chart(data[:10])
# st.map(data[:10])
st.scatter_chart(data[:10])

# Add sidebar
a = st.sidebar.radio('Select one:', [1, 2])
st.sidebar.caption("This is a cool caption")
st.sidebar.image('./smile.png')

# Add columns
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")