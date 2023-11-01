from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2

df = clean()
print(df.shape)
print(df.columns)
y = df['target'].values.astype(float)
print(y.shape)
x_df = df.drop(columns = 'target')
x = x_df.values.astype(float)
print(x.shape)
print(type(y))
print(type(x))
print(type(x.shape))
custom_adam = Adam(learning_rate=0.001)
model = Sequential()
model.add(Dense(14, input_shape=(x.shape[1], ), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(6, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(3, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer=custom_adam, metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=32, validation_split = 0.2)
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))
