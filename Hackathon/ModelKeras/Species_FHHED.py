import os
import pandas as pd
path_input_data = os.path.join('.', 'test_combined_Species.csv')

df_data = pd.read_csv(path_input_data, delimiter=',')
df_data.drop(df_data.columns[0], axis=1, inplace=True)

##########################
# You do the preprocessing here
##########################

path_output_data = path_input_data
df_data.to_csv(path_output_data)
