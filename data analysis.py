import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 caltech.csv")
df2 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 jpl.csv")
df3 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 office1.csv")

frequency = df1['userID'].value_counts()
print(frequency)

frequency2 = df2['userID'].value_counts()
print(frequency2)

frequency3 = df3['userID'].value_counts()
print(frequency3)

frequency.plot(kind="bar")

plt.show()




