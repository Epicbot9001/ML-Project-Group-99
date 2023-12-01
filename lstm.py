from cleaning import clean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam

def custom_metric(y_true, y_pred):
    threshold = 0.5
    absolute_difference = K.abs((y_true - y_pred) / y_true)
    correct_predictions = K.cast(absolute_difference <= threshold, dtype='float32')
    accuracy = K.mean(correct_predictions)
    accuracy *= 100
    return accuracy

df = clean()
df['stock_id'] = df['stock_id'].astype('category')

# List to store individual models
models = []

iters = 0
max_iters = 2

for stock_id in df['stock_id'].cat.categories:
    if iters >=  max_iters:
        break

    print("\nstock id", iters)

    stock_data = df[df['stock_id'] == stock_id].drop(columns=['target', 'stock_id'])

    # Split data into X and y
    X = stock_data.values
    y = df[df['stock_id'] == stock_id]['target'].values

    # Reshape X to 3D for LSTM
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Build model
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(1, X.shape[2]), return_sequences=True))  # Experiment with return_sequences
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32))  # Adding another LSTM layer
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))  # Output layer for regression


    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=["mae", custom_metric],
    )

    # Fit the model
    model.fit(X, y, epochs=10, batch_size=8, validation_split=0.1)
    
    # Append the trained model to the list
    models.append(model)

    iters += 1

# Now you have a list of models, each trained on a specific stock_id
# You can use these models for prediction on new data


'''from cleaning import clean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

def custom_metric(y_true, y_pred):
    threshold = 0.5
    absolute_difference = K.abs((y_true - y_pred)/y_true)
    correct_predictions = K.cast(absolute_difference <= threshold, dtype='float32')
    accuracy = K.mean(correct_predictions)
    accuracy *= 100
    return accuracy

df = clean()
y = df['target'].values.astype(float)
x_df = df.drop(columns='target')
df['stock_id'] = df['stock_id'].astype('category')

grouped_data = x_df.groupby('stock_id')
i = 0

x_sequences = []
y_labels = []

for stock_id, group in grouped_data:
    stock_data = group.drop('stock_id', axis=1).values
    x_sequences.append(stock_data)
    
    # Assuming you want to predict the target value for the last time step
    y_labels.append(df[df['stock_id'] == stock_id]['target'].values[-1])

max_sequence_length = max(arr.shape[0] for arr in x_sequences)
common_shape = (max_sequence_length, x_sequences[0].shape[1])

# Padding input sequences
padded_x_sequences = [np.pad(arr, ((0, max_sequence_length - arr.shape[0]), (0, 0)), mode='constant') for arr in x_sequences]
x = np.stack(padded_x_sequences, axis=0)

# Converting y_labels to a numpy array
y = np.array(y_labels)

model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(max_sequence_length, x_sequences[0].shape[1])))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))  # Output layer for regression

model.compile(
    loss='mean_squared_error',  # Use mean squared error for regression
    optimizer="sgd",
    metrics=["mae", custom_metric],
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.1)
'''