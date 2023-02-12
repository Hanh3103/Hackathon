import os
import pandas as pd
import pickle as pk

path_input_data = os.path.join('.', 'train_combined_Species.csv')

df = pd.read_csv(path_input_data, delimiter=',')
df.drop(df.columns[0], axis=1, inplace=True)

##########################

outputvalues = df['label']
y = outputvalues.values

y = pd.DataFrame(y)
##########################

#path_output_data = path_input_data
#df.to_csv("test.csv", index=False, header=False)
y.to_csv("y.csv", index=False)