
from pathlib import Path
from create_metafeatures import calculate_metafeatures
import torch
from model import MetallicDL
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import pickle
import pandas as pd

# Setup our data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# DEEP LEARNING MODEL
# Train a model
model = MetallicDL()
model.save('model.pth')

# Load the model
model = None
model = MetallicDL.load('model.pth')

# Make a dummy prediction
print(model.predict(torch.rand(64)).detach().numpy())


# SKLEARN MODEL
# Train a MODEL
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
model = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make a dummy prediction
print(model.predict([X_test[0]]))



# CREATING META FEATURES
# saved_path = Path('user_df.csv')
# user_df = pd.read_csv('../data/processed_datasets/Australian.csv')
# user_df.to_csv(saved_path, index=False)
# meta_features = calculate_metafeatures(saved_path)

# This will be passed into the model.
# I have not implemented this yet.
# I will provide more information soon about that.
# You need the rest of the github repo to run this commented out code.
