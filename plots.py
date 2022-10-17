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

#most important looking: V18, V17, V16, V14, a lot of them are similar but above does not 18 look very useful
#might be useful to look at the amounts in more detail (not through plots)

#correlation coeficients:
cor = df.corr()
print(cor["Class"])

#returns 
# Time     -0.012323
# V1       -0.101347
# V2        0.091289
# V3       -0.192961 #strong
# V4        0.133447
# V5       -0.094974
# V6       -0.043643
# V7       -0.187257 #strong
# V8        0.019875
# V9       -0.097733
# V10      -0.216883 #strong
# V11       0.154876
# V12      -0.260593 #strong
# V13      -0.004570
# V14      -0.302544 #strong
# V15      -0.004223
# V16      -0.196539 #strong
# V17      -0.326481 #strongest
# V18      -0.111485
# V19       0.034783
# V20       0.020090
# V21       0.040413
# V22       0.000805
# V23      -0.002685
# V24      -0.007221
# V25       0.003308
# V26       0.004455
# V27       0.017580
# V28       0.009536
# Amount    0.005632
# Class     1.000000

#kind of confirms what i saw visually