import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

df = pd.read_csv("creditcard.csv") #dataset with creditcard transactions, some of which are fraudulant. Want to predict, whether they are or not.

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

classes = test_df["Class"].to_numpy()

def count_correct(classes, credences, cutoff):
    n_correct_predictions = 0
    n_false_positive = 0
    n_false_negative = 0
    for i, p in enumerate(credences):
        binary_prediction = p > cutoff
        if binary_prediction == (classes[i] == 1): #if prediciton (given cutoff) and actual value match
            n_correct_predictions += 1
        #else:
        #    if 
    
    return n_correct_predictions

def plot_cutoff_effectiveness(cutoffs):
    n_correct = [count_correct(classes, credences, cutoff) for cutoff in cutoffs]
    return plt.plot(cutoffs, n_correct)

plot_cutoffs = np.concatenate((np.arange(0.04, step=0.001), np.arange(start=0.04, stop=0.5, step=0.05)))
plot_cutoff_effectiveness(plot_cutoffs) #one false categorization (119 correct ones) for cutoff values of about  0.01 to 0.022

plt.show()

cutoff = 0.15
predictions = []
for i, row in test_df.iterrows():
    if row["Credence"] > cutoff:
        predictions.append(1)
    else:
        predictions.append(0)

test_df["Prediction"] = predictions

