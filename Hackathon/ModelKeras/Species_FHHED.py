import os
import pandas as pd
import pickle as pk

path_input_data = os.path.join('.', 'train_combined_Species.csv')

df = pd.read_csv(path_input_data, delimiter=',')
df.drop(df.columns[0], axis=1, inplace=True)

##########################
df = df.fillna(0)

inputvalues = df.drop(['label'], axis=1)
outputvalues = df['label']
X, y = inputvalues.values, outputvalues.values
X = X.astype('float32')

scaler = pk.load(open("scaler.pkl",'rb'))
X = scaler.transform(X)

pca = pk.load(open("pca.pkl",'rb'))
X = pca.transform(X)

X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=['label'])
df = pd.concat([X, y], axis=1)
print(df)
##########################

#path_output_data = path_input_data
#df.to_csv(path_output_data)
