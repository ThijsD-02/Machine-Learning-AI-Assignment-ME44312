import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Bestand inlezen
df1 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 office1.csv")
df2 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 jpl.csv")
df3 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 caltech.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

# Bekijken hoeveel niet-lege waarden er zijn en de eerste paar waarden tonen
print(df["userInputs/0/requestedDeparture"].dropna().head())

df["disconnectTime"] = pd.to_datetime(df["disconnectTime"])
df["userInputs/0/requestedDeparture"] = pd.to_datetime(df["userInputs/0/requestedDeparture"])

df["time_difference"] = (df["disconnectTime"] - df["userInputs/0/requestedDeparture"]).dt.total_seconds() / 60
df = df[df["time_difference"] <= 1500]
df = df[df["time_difference"] >= -1500]

mean_time_diff = df["time_difference"].mean()
std_time_diff = df["time_difference"].std()

print(f"Mean time difference: {mean_time_diff:.2f} minutes")
print(f"Standard deviation: {std_time_diff:.2f} minutes")

# Genereer een reeks x-waarden (bijv. van minimum tot maximum van de tijdsverschillen)
xmin, xmax = df["time_difference"].min(), df["time_difference"].max()
x = np.linspace(xmin, xmax, 100)

# Genereer de bijbehorende y-waarden van de normaalverdeling
y = stats.norm.pdf(x, mean_time_diff, std_time_diff)

# Plot het histogram van de tijdsverschillen
plt.figure(figsize=(10, 5))
df["time_difference"].dropna().hist(bins=50, edgecolor='black', density=True, alpha=0.6)

# Plot de normaalverdeling
plt.plot(x, y, 'r-', label='Normal distribution', linewidth=2)

plt.text(xmin + 0.55 * (xmax - xmin), max(y) * 0.85, 
         f"$\mu = {mean_time_diff:.2f}$\n$\sigma = {std_time_diff:.2f}$", 
         fontsize=12, color='black')

# Voeg labels en titel toe
plt.xlabel("Difference in minutes")
plt.ylabel("Number of sessions")
plt.title("Difference between disconnection time and requested departure")
plt.legend()

# Toon de plot
plt.show()