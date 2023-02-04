### Random Forest Classifier ###

# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

# import dataset
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'train_combined_Species.csv')
df = pd.read_csv(filename, delimiter = ',')

# Drop the index column
df.drop(df.columns[0], axis=1, inplace=True)

# Selecting features X and target Y
X = df.loc[:, df.columns != "label"] # all columns except of the label column
Y = df["label"]

# Splitting the dataset into train and test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Feature Scaling with MinMaxScaler
scaler = MinMaxScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

# fitting classifier into training set
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 1, max_depth= None, min_samples_split = 5)
classifier.fit(X_Train,Y_Train)

# predicting test set results
Y_Pred = classifier.predict(X_Test)

# Print Classification Report
print("Classification Report for Random Forest Classifier: \n {}\n".format(metrics.classification_report(Y_Test, Y_Pred)))