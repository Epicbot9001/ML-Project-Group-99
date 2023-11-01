from cleaning import clean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = clean()
print(df.shape)
print(df.columns)
y = df['target']
print(y.shape)
x_df = df.drop(columns = 'target')
x = x_df.values
model = Sequential()
model.add(Dense(15, input_shape=(4804764, 15), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))
