from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
model = Sequential()
model.add(Dense(14, input_shape=(x.shape[1], ), activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))
