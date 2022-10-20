#Plotting whether transactions are fraudulent or not as a function of all of the variables

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

grouped_df = df.groupby("Class")
frauds_df = grouped_df.get_group(1)

samples_df = grouped_df.get_group(0).sample(1000)

combined_df = pd.concat([samples_df, frauds_df]) #dataframe with all fraudulent and randomly sampled non-fraudulent data (because there are so little fraudulent ones)



#plot the category as a function of all of the variables

vars = list(df.columns)
vars.remove("Class")

for var in vars:
    pass
    combined_df.plot(x=var, y="Class", kind="scatter")

plt.show()

#correlation coeficients:
cor = df.corr()
print(cor["Class"])