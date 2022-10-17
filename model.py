import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("creditcard.csv")

grouped_df = df.groupby("Class")
frauds_df = grouped_df.get_group(1)
valid_df = grouped_df.get_group(0)

s_frauds = frauds_df.sample(20)
s_valid = valid_df.sample(100)

test_df = pd.concat([s_frauds, s_valid]) #test data

df.drop(test_df.index) #df now should contain training data (all data that is not test data)


vars = list(df.columns)
vars.remove("Class")

regr = linear_model.LinearRegression()
regr.fit(df[vars], df["Class"])

print(regr.coef_)