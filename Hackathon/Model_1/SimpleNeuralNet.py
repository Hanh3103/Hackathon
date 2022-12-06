import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
# load the dataset
mapping = [3,4,2,1]
df = [pd.read_csv('../HackathonData/CAD/ClassCAD_train.csv'), pd.read_csv('../HackathonData/CKD/ClassCKD_train.csv'),
      pd.read_csv('../HackathonData/IBD/ClassIBD_train.csv'), pd.read_csv('../HackathonData/T2D/ClassT2D_train.csv')]
df = pd.concat([x if x['label'].replace(1, mapping[idx], inplace=True) == True else x for idx, x in enumerate(df)], join="outer", axis=0)
# split into input and output columns
inputValues = df.drop(['label'], axis=1)
outputValues = df['label']
X, y = inputValues.values, outputValues.values
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(5000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(5000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(500, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(200, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(4, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=1000, batch_size=572, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)


A = df.drop('sample_ID', axis=1)
I1 = A[A.label == 1]
countright = 0
for index, row in I1.iterrows():
    tmp = row[: -1]
    tmp = tmp[:,None]
    tmp = np.transpose(tmp)
    #print(tmp.values)
    yhat = model.predict([tmp])
    if row[-1:].values == yhat:
        countright+=1
    print('Predicted: %.3f' % yhat)

print(countright)
print(len(I1))
print(countright / len(I1))
