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

#frequency.plot(kind="bar")

plt.show()

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


#Arrivel time vs. staying duration

def process_data3(df):
    # Convert to datetime (ensuring timezone awareness)
    df.loc[:, 'connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)

    # Convert connectionTime to PDT
    df.loc[:, 'connectionTime_PDT'] = df['connectionTime'].dt.tz_convert('America/Los_Angeles')

    # Remove timezone from connectionTime_PDT for plotting purposes
    df.loc[:, 'arrivalMinutes'] = df['connectionTime_PDT'].dt.hour * 60 + df['connectionTime_PDT'].dt.minute  

    # Convert disconnectTime to PDT and make it timezone-naive for calculations

    # Calculate staying duration

    return df.dropna(subset=['arrivalMinutes', 'stayingDuration'])

# Process all datasets
df1, df2, df3 = [process_data3(df) for df in [df1, df2, df3]]

# Plot settings
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plots for each dataset
ax.scatter(df1['arrivalMinutes'], df1['stayingDuration'], color='b', s=10, alpha=0.7, label="Caltech")
ax.scatter(df2['arrivalMinutes'], df2['stayingDuration'], color='r', s=10, alpha=0.7, label="JPL")
ax.scatter(df3['arrivalMinutes'], df3['stayingDuration'], color='g', s=10, alpha=0.7, label="Office 1")

# Format x-axis to show time labels
ax.set_xticks(range(0, 1441, 120))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
ax.set_ylim(0, 1000)

# Format plot
ax.set_xlabel('Arrival Time (PDT)')
ax.set_ylabel('Staying Duration (Minutes)')
ax.set_title('Arrival Time vs. Staying Duration (PDT)')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


# Arrival time vs kwh delivered
# # Function to process datasets
# def process_data2(df):
#     df.loc[:, 'connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)  # Convert to UTC
#     df.loc[:, 'connectionTime_PDT'] = df['connectionTime'].dt.tz_convert('America/Los_Angeles')  # Convert to PDT
#     df.loc[:, 'arrivalMinutes'] = df['connectionTime_PDT'].dt.hour * 60 + df['connectionTime_PDT'].dt.minute  # Convert to minutes since midnight
    
#      # Debug: Print a few timestamps before and after conversion
#     print("Original UTC time:", df['connectionTime'].head(3))
#     print("Converted PDT time:", df['connectionTime_PDT'].head(3))

#     return df.dropna(subset=['arrivalMinutes', 'kWhDelivered'])  # Drop missing values

# # Process all datasets
# df1, df2, df3 = [process_data2(df) for df in [df1, df2, df3]]

# # Plot settings
# fig, ax = plt.subplots(figsize=(12, 6))

# # Scatter plots for each dataset
# ax.scatter(df1['arrivalMinutes'], df1['kWhDelivered'], color='b', s=10, alpha=0.7, label="Caltech")
# ax.scatter(df2['arrivalMinutes'], df2['kWhDelivered'], color='r', s=10, alpha=0.7, label="JPL")
# ax.scatter(df3['arrivalMinutes'], df3['kWhDelivered'], color='g', s=10, alpha=0.7, label="Office 1")

# # Format x-axis to show time labels
# ax.set_xticks(range(0, 1441, 120))  # Set x-ticks every 2 hours
# ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])  # Convert to readable format

# # Format plot
# ax.set_xlabel('Arrival Time (PDT)')
# ax.set_ylabel('kWh Delivered')
# ax.set_title('Arrival Time vs. kWh Delivered (PDT Time)')
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.5)

# # Show the plot
# plt.show()



# plot arrival date vs. kwh delivered
# # Function to process datasets
# def process_data1(df):
#     df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce')
#     df = df.dropna(subset=['connectionTime', 'kWhDelivered'])  # Drop missing values
#     return df

# # Process all datasets
# df1, df2, df3 = [process_data1(df) for df in [df1, df2, df3]]

# # Plot settings
# fig, ax = plt.subplots(figsize=(12, 6))

# # Scatter plots for each dataset
# ax.scatter(df1['connectionTime'], df1['kWhDelivered'], color='b', s=5, alpha=0.7, label="Caltech")
# ax.scatter(df2['connectionTime'], df2['kWhDelivered'], color='r', s=5, alpha=0.7, label="JPL")
# ax.scatter(df3['connectionTime'], df3['kWhDelivered'], color='g', s=5, alpha=0.7, label="Office 1")

# # Format plot
# ax.set_xlabel('Arrival Time (Connection Time)')
# ax.set_ylabel('kWh Delivered')
# ax.set_title('Arrival Time vs. kWh Delivered')
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.5)
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# # Show the plot
# plt.show()




##Plots of durations vs. kwh
# # Create scatter plot
# plt.figure(figsize=(12, 6))
# plt.scatter(df1['chargingDuration'], df1['kWhDelivered'], color='b', s=10, alpha=0.7, label="Caltech")
# plt.scatter(df2['chargingDuration'], df2['kWhDelivered'], color='r', s=10, alpha=0.7, label="JPL")
# plt.scatter(df3['chargingDuration'], df3['kWhDelivered'], color='g', s=10, alpha=0.7, label="Office 1")

# # Set limits and labels
# plt.xlim(0, 2000)  # Limit x-axis to 2000 minutes
# plt.xlabel('Charging Duration (Minutes)')
# plt.ylabel('kWh Delivered')
# plt.title('Charging Duration vs kWh Delivered')

# #Add grid and legend
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# #Show the plot
# plt.show()

# # Create scatter plot
# plt.figure(figsize=(12, 6))
# plt.scatter(df1['stayingDuration'], df1['kWhDelivered'], color='lightblue', s=10, alpha=0.7, label="Caltech stay")
# plt.scatter(df2['stayingDuration'], df2['kWhDelivered'], color='orange', s=10, alpha=0.7, label="JPL stay")
# plt.scatter(df3['stayingDuration'], df3['kWhDelivered'], color='lightgreen', s=10, alpha=0.7, label="Office 1 stay")

# # Set limits and labels
# plt.xlim(0, 4000)  # Limit x-axis to 2000 minutes
# plt.xlabel('Staying Duration (Minutes)')
# plt.ylabel('kWh Delivered')
# plt.title('Staying Duration vs kWh Delivered')

# # Add grid and legend
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# # Show the plot
# plt.show()

# # Create scatter plot
# plt.figure(figsize=(12, 6))
# plt.scatter(df1['nonuseDuration'], df1['kWhDelivered'], color='darkblue', s=10, alpha=0.7, label="Caltech non use")
# plt.scatter(df2['nonuseDuration'], df2['kWhDelivered'], color='yellow', s=10, alpha=0.7, label="JPL non use")
# plt.scatter(df3['nonuseDuration'], df3['kWhDelivered'], color='darkgreen', s=10, alpha=0.7, label="Office 1 non use")

# # Set limits and labels
# plt.xlim(0, 2000)  # Limit x-axis to 2000 minutes
# plt.xlabel('Non-use Duration (Minutes)')
# plt.ylabel('kWh Delivered')
# plt.title('Non-use Duration vs kWh Delivered')

# # Add grid and legend
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# # Show the plot
# plt.show()
