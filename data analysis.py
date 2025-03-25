import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
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




=======
# Load the CSV into a DataFrame
df = pd.read_csv('Data/acndata_sessions 2019_11 till 2020_5 caltech.csv')

# Load the three CSV files
df1 = pd.read_csv('Data/acndata_sessions 2019_11 till 2020_5 caltech.csv')
df2 = pd.read_csv('Data/acndata_sessions 2019_11 till 2020_5 jpl.csv')
df3 = pd.read_csv('Data/acndata_sessions 2019_11 till 2020_5 office1.csv')

# Function to process each dataset
def process_data(df):
    df[['connectionTime', 'doneChargingTime', 'disconnectTime']] = df[
        ['connectionTime', 'doneChargingTime', 'disconnectTime']
    ].apply(pd.to_datetime, errors='coerce')

    df['chargingDuration'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds() / 60
    df['stayingDuration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 60
    df['nonuseDuration'] = (df['stayingDuration'] - df['chargingDuration'])
    return df.dropna(subset=['chargingDuration', 'stayingDuration', 'nonuseDuration', 'kWhDelivered'])

# Process all datasets efficiently
datasets = [df1, df2, df3]
df1, df2, df3 = [process_data(df) for df in datasets]

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(df1['chargingDuration'], df1['kWhDelivered'], color='b', s=10, alpha=0.7, label="Caltech")
plt.scatter(df2['chargingDuration'], df2['kWhDelivered'], color='r', s=10, alpha=0.7, label="JPL")
plt.scatter(df3['chargingDuration'], df3['kWhDelivered'], color='g', s=10, alpha=0.7, label="Office 1")

# Set limits and labels
plt.xlim(0, 2000)  # Limit x-axis to 2000 minutes
plt.xlabel('Charging Duration (Minutes)')
plt.ylabel('kWh Delivered')
plt.title('Charging Duration vs kWh Delivered')

#Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

#Show the plot
plt.show()

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(df1['stayingDuration'], df1['kWhDelivered'], color='lightblue', s=10, alpha=0.7, label="Caltech stay")
plt.scatter(df2['stayingDuration'], df2['kWhDelivered'], color='orange', s=10, alpha=0.7, label="JPL stay")
plt.scatter(df3['stayingDuration'], df3['kWhDelivered'], color='lightgreen', s=10, alpha=0.7, label="Office 1 stay")

# Set limits and labels
plt.xlim(0, 4000)  # Limit x-axis to 2000 minutes
plt.xlabel('Staying Duration (Minutes)')
plt.ylabel('kWh Delivered')
plt.title('Staying Duration vs kWh Delivered')

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.show()

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(df1['nonuseDuration'], df1['kWhDelivered'], color='darkblue', s=10, alpha=0.7, label="Caltech non use")
plt.scatter(df2['nonuseDuration'], df2['kWhDelivered'], color='yellow', s=10, alpha=0.7, label="JPL non use")
plt.scatter(df3['nonuseDuration'], df3['kWhDelivered'], color='darkgreen', s=10, alpha=0.7, label="Office 1 non use")

# Set limits and labels
plt.xlim(0, 2000)  # Limit x-axis to 2000 minutes
plt.xlabel('Non-use Duration (Minutes)')
plt.ylabel('kWh Delivered')
plt.title('Non-use Duration vs kWh Delivered')

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.show()
>>>>>>> b828a1128c53aa1b901d1e0a1368da4f843b6628
