import pandas as pd
import numpy as np
housing = pd.read_csv("data.csv")

# Train, Test, Splitting
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Splitting in Different Sets
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Adding Missing Attributes using Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns) 

# Creating a pipeline and doing feature scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

# Selecting A Desired Model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# Saving The Model
# from joblib import dump, load
# dump(model, 'Price.joblib') 


# Using The Model
from joblib import dump,load
model = load('Price.joblib') 
temp_data=np.array([[1.23247,0,8.14,0,0.538,6.142,91.7,3.9769,4,307,21,396.9,18.72]])

final_data= pd.DataFrame(temp_data, columns=housing.columns)   #creating dataframe of array
features=my_pipeline.transform(final_data)   #transforming it sing our pipline

final_price=model.predict(features)[0]
print(final_price)