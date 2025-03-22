import pickle

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from sklearn import preprocessing

# Load data
print("Loading data...")
df = pd.read_csv("./data/data_simplified.csv")

# Preprocess data
encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

def preprocess_column(col):
    if col.dtype == 'object' or col.dtype == 'category':
        return encoder.fit_transform(col)
    elif col.dtype == 'bool':
        return col.astype('int')
    elif col.nunique() <= 6:
        return col
    else:
        return discretizer.fit_transform(col.values.reshape(-1, 1)).astype(int).flatten()

print("Preprocessing data...")
df = df.apply(preprocess_column)

# optimizing memory
print("Converting data types...")
df = df.astype({col: 'int8' for col in df.select_dtypes('int64').columns})  # Convert large integers

# Define structure
print("Creating structure...")
var_names = df.columns.tolist()
mnts = [var for var in var_names if "Mnt" in var]

edges = {
    "Income": ["AcceptedCmps", "TotPurchases", "NumDealsPurchases", *mnts],
    "Age": ["AcceptedCmps", "TotPurchases", "NumDealsPurchases", *mnts],
    "Customer_Days": ["TotPurchases", "NumDealsPurchases", "Complain"],
    "Complain": ["Response"],
    "AcceptedCmps": ["Response"],
    "TotPurchases": ["Response"],
    "NumDealsPurchases": ["Response"],
    "Recency": ["Response"]
}
edges.update({mnt: ["AcceptedCmps", "Recency"] for mnt in mnts})
edges_tuples = [(n1, n2) for n1, nodes in edges.items() for n2 in nodes]

# Create Bayesian Network
print("Creating Bayesian Network...")
model = BayesianNetwork(edges_tuples)

# Fit parameters using Maximum Likelihood Estimation
print("Training model...")
model.fit(df, estimator=MaximumLikelihoodEstimator)

# check the model after training
print("Checking model...")
model.check_model()

# Save the model
print("Saving model...")
with open("./data/bn.pkl", "wb") as f:
    pickle.dump(model, f)
