from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=32)
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))
