# from cleaning import clean
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from keras import backend as K
# import plotly.graph_objs as go
# import plotly.express as px

# def custom_accuracy(y_true, y_pred, threshold=1.0):
#     within_threshold = y_pred * y_true > 0
#     return K.mean(within_threshold)

# df = clean()
# y = df['target'].values.astype(float)
# x_df = df.drop(columns='target')
# x = x_df.values.astype(float)

# custom_adam = Adam(learning_rate=0.001)
# model = Sequential()
# model.add(Dense(14, input_shape=(x.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(6, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(3, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer=custom_adam, metrics=[custom_accuracy, 'mean_absolute_error'])

# loss_values = []
# val_loss_values = []
# accuracy_values = []
# val_accuracy_values = []

# for epoch in range(10):
#     history = model.fit(x, y, epochs=1, batch_size=256, validation_split=0.1, verbose=0)
#     loss_values.append(history.history['loss'][0])
#     val_loss_values.append(history.history['val_loss'][0])
#     accuracy_values.append(history.history['custom_accuracy'][0])
#     val_accuracy_values.append(history.history['val_custom_accuracy'][0])

# # Plotting the metrics
# fig_loss = go.Figure()
# fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_values) + 1)), y=loss_values, mode='lines', name='Training Loss'))
# fig_loss.add_trace(go.Scatter(x=list(range(1, len(val_loss_values) + 1)), y=val_loss_values, mode='lines', name='Validation Loss'))
# fig_loss.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')

# fig_accuracy = go.Figure()
# fig_accuracy.add_trace(go.Scatter(x=list(range(1, len(accuracy_values) + 1)), y=accuracy_values, mode='lines', name='Training Accuracy'))
# fig_accuracy.add_trace(go.Scatter(x=list(range(1, len(val_accuracy_values) + 1)), y=val_accuracy_values, mode='lines', name='Validation Accuracy'))
# fig_accuracy.update_layout(title='Accuracy Over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')

# fig_loss.show()
# fig_accuracy.show()


from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras import backend as K
import plotly.graph_objs as go

def custom_accuracy(y_true, y_pred, threshold=1.0):
    within_threshold = y_pred * y_true > 0
    return K.mean(within_threshold)

df = clean()
y = df['target'].values.astype(float)
x_df = df.drop(columns='target')
x = x_df.values.astype(float)

custom_adam = Adam(learning_rate=0.001)
model = Sequential()
model.add(Dense(14, input_shape=(x.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(6, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(3, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=custom_adam, metrics=[custom_accuracy, 'mean_absolute_error'])

loss_values = []
val_loss_values = []
accuracy_values = []
val_accuracy_values = []
mae_values = []
val_mae_values = []

for epoch in range(10):
    history = model.fit(x, y, epochs=1, batch_size=256, validation_split=0.1, verbose=0)
    loss_values.append(history.history['loss'][0])
    val_loss_values.append(history.history['val_loss'][0])
    accuracy_values.append(history.history['custom_accuracy'][0])
    val_accuracy_values.append(history.history['val_custom_accuracy'][0])
    
    y_pred = model.predict(x)
    mae = K.eval(K.mean(K.abs(y - y_pred)))
    mae_values.append(mae)

    y_val_pred = model.predict(x[int(len(x) * 0.9):])
    val_mae = K.eval(K.mean(K.abs(y[int(len(y) * 0.9):] - y_val_pred)))
    val_mae_values.append(val_mae)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_values) + 1)), y=loss_values, mode='lines', name='Training Loss'))
fig_loss.add_trace(go.Scatter(x=list(range(1, len(val_loss_values) + 1)), y=val_loss_values, mode='lines', name='Validation Loss'))
fig_loss.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')

fig_accuracy = go.Figure()
fig_accuracy.add_trace(go.Scatter(x=list(range(1, len(accuracy_values) + 1)), y=accuracy_values, mode='lines', name='Training Accuracy'))
fig_accuracy.add_trace(go.Scatter(x=list(range(1, len(val_accuracy_values) + 1)), y=val_accuracy_values, mode='lines', name='Validation Accuracy'))
fig_accuracy.update_layout(title='Accuracy Over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')

fig_mae = go.Figure()
fig_mae.add_trace(go.Scatter(x=list(range(1, len(mae_values) + 1)), y=mae_values, mode='lines', name='Training MAE'))
fig_mae.add_trace(go.Scatter(x=list(range(1, len(val_mae_values) + 1)), y=val_mae_values, mode='lines', name='Validation MAE'))
fig_mae.update_layout(title='Mean Absolute Error (MAE) Over Epochs', xaxis_title='Epoch', yaxis_title='MAE')

fig_loss.show()
fig_accuracy.show()
fig_mae.show()
