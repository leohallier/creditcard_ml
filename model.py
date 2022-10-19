import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("creditcard.csv")

#generate test set, in which fraudulent cases are overrepresented (since there are so few of them)
grouped_df = df.groupby("Class")
frauds_df = grouped_df.get_group(1)
valid_df = grouped_df.get_group(0)

s_frauds = frauds_df.sample(20)
s_valid = valid_df.sample(100)

test_df = pd.concat([s_frauds, s_valid]) #test data set with 20 fraudulent and 100 legitimate transactions

df.drop(test_df.index) #df now should contain training data (all data that is not test data)


vars = list(df.columns)
vars.remove("Class")

regr = linear_model.LinearRegression()
regr.fit(df[vars], df["Class"])

#print(regr.coef_)

credences = regr.predict(test_df[vars])

test_df["Credence"] = credences

test_df.plot(x="Credence", y="Class", kind="scatter") #plot shows really good separation at a cutoff of about 0.02
plt.show()

cutoff = 0.02
predictions = []
for i, row in test_df.iterrows():
    if row["Credence"] > cutoff:
        predictions.append(1)
    else:
        predictions.append(0)

test_df["Prediction"] = predictions

def count_correct(cutoff):
    n_correct_predictions = 0
    for i, p in enumerate(predictions):
        binary_prediction = p > cutoff
        if binary_prediction and (test_df.iloc[i]["Class"] == 1): #if prediciton (given cutoff) and actual value match
            n_correct_predictions += 1
    
    return n_correct_predictions