import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

grouped_df = df.groupby("Class")
frauds_df = grouped_df.get_group(1)

samples_df = grouped_df.get_group(0).sample(1000)

combined_df = pd.concat([samples_df, frauds_df])

combined_df.plot(x="V1", y="Class", kind="scatter")

plt.show()