from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import plotly.graph_objects as go
import numpy as np

def custom_metric(y_true, y_pred):
    # Define a threshold (e.g., 0.1)
    threshold = 0.5

    # Calculate the absolute difference between true and predicted values
    absolute_difference = K.abs((y_true - y_pred)/y_true)

    # Create a binary mask indicating whether the absolute difference is within the threshold
    correct_predictions = K.cast(absolute_difference <= threshold, dtype='float32')

    # Calculate the mean of the binary mask to get the accuracy
    accuracy = K.mean(correct_predictions)
    accuracy *= 100

    return accuracy

df = clean()
print(df.shape)
print(df.columns)
print(df.head())
y = df['target'].values.astype(float)
print(y.shape)
x_df = df.drop(columns = 'target')
x = x_df.values.astype(float)
print(x.shape)
print(type(y))
print(type(x))
print(type(x.shape))
model = Sequential()
model.add(Dense(512, input_shape=(x.shape[1], ), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=["mae", custom_metric])
# Train the model and save the history
history = model.fit(x, y, epochs=5, batch_size=16384, validation_split=0.1)

# Create a Plotly figure
fig_loss = go.Figure()
fig_metrics = go.Figure()

# Plot training loss
fig_loss.add_trace(go.Scatter(x=np.arange(1, len(history.history['loss']) + 1), y=history.history['loss'], mode='lines', name='Training Loss'))

# Plot validation loss if available
if 'val_loss' in history.history:
    fig_loss.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], mode='lines', name='Validation Loss'))

# Plot mae metric
fig_metrics.add_trace(go.Scatter(x=np.arange(1, len(history.history['mae']) + 1), y=history.history['mae'], mode='lines', name='MAE'))

# Plot validation mae metric if available
if 'mae_loss' in history.history:
    fig_metrics.add_trace(go.Scatter(x=np.arange(1, len(history.history['mae_loss']) + 1), y=history.history['mae_loss'], mode='lines', name='MAE Loss'))

# Plot custom metric
fig_metrics.add_trace(go.Scatter(x=np.arange(1, len(history.history['custom_metric']) + 1), y=history.history['custom_metric'], mode='lines', name='Custom Metric'))

# Plot validation custom metric if available
if 'val_custom_metric' in history.history:
    fig_metrics.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_custom_metric']) + 1), y=history.history['val_custom_metric'], mode='lines', name='Validation Custom Metric'))

# Update layout
fig_metrics.update_layout(title='Training Metrics',
                  xaxis_title='Epoch',
                  yaxis_title='Metric Value',
                  legend=dict(x=0, y=1),
                  template='plotly_dark')

# Show the figure
#fig.write_html("training_metrics.html")
fig_loss.show()
fig_metrics.show()