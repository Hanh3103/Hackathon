import os
import pandas as pd
import pickle as pk

path_input_data = os.path.join('.', 'test_combined_Species.csv')

df = pd.read_csv(path_input_data, delimiter=',')
df.drop(df.columns[0], axis=1, inplace=True)

##########################
df = df.fillna(0)

X = df.values
X = X.astype('float32')

scaler = pk.load(open("scaler.pkl",'rb'))
X = scaler.transform(X)

pca = pk.load(open("pca.pkl",'rb'))
X = pca.transform(X)

X = pd.DataFrame(X)
df = X
##########################

path_output_data = path_input_data
# df.to_csv(path_output_data)
# it should be this
df.to_csv(path_output_data, index=False)
