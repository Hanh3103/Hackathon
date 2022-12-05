import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('TkAgg')
sns.set(style="darkgrid")

df = pd.read_csv('./HackathonData/CAD/ClassCAD_train.csv')
list_attributes = list(df.columns)
# print(data.head(3))

# Formatting the data
healthy_data = df[df['label'] == 0]
not_healthy_data = df[df['label'] == 1]
# print(healthy_data)
# print(not_healthy_data)

# Plotting
# sns.lmplot(x="Bacteria;Abditibacteriota;Abditibacteria", y="Bacteria;Acidobacteriota;Aminicenantia", hue="label",
# data=con)
# sns.kdeplot(x='Bacteria;Abditibacteriota;Abditibacteria', hue='label', data=data)
df_drop_first_columns = df.iloc[:, 1:]
df_long = pd.melt(df_drop_first_columns, "label", var_name="Bacteria", value_name="Number")
# print(df_long.head(786).to_string())
sns.boxplot(x="Bacteria", hue="label", y="Number", data=df_long)
plt.show()
# print(con.columns)
# print(list_attributes)
#df.describe()

