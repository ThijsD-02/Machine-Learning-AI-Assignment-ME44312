'''
This script produces plots to analyse data from the diffent locations and are discussed in section 3 of the paper.

The following plots where made: 
- Arrival Time vs. Staying Duration
- Charging Duration vs kWh Delivered 
- Staying Duration vs kWh Delivered
- Non-use Duration vs kWh Delivered
- Arrival Time vs kWh Delivered
- Arrival Date vs kWh Delivered
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the CSV files
df1 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 caltech.csv")
df2 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 jpl.csv")
df3 = pd.read_csv("Data/acndata_sessions 2019_11 till 2020_5 office1.csv")

# Function to process data
def process_data(df):
    # Convert to datetime
    df[['connectionTime', 'doneChargingTime', 'disconnectTime']] = df[
        ['connectionTime', 'doneChargingTime', 'disconnectTime']
    ].apply(pd.to_datetime, errors='coerce')

    # Calculate durations
    df['chargingDuration'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds() / 60
    df['stayingDuration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 60
    df['nonuseDuration'] = df['stayingDuration'] - df['chargingDuration']
    
    # Convert connectionTime to PDT
    df['connectionTime_PDT'] = df['connectionTime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
    df['arrivalMinutes'] = df['connectionTime_PDT'].dt.hour * 60 + df['connectionTime_PDT'].dt.minute
    df['arrivalDate'] = df['connectionTime_PDT'].dt.date  # Extract arrival date

    return df.dropna(subset=['chargingDuration', 'stayingDuration', 'nonuseDuration', 'kWhDelivered', 'arrivalMinutes'])

# Process all datasets
df1, df2, df3 = [process_data(df) for df in [df1, df2, df3]]

# Define colors
colors = {'Caltech': 'blue', 'JPL': 'red', 'Office1': 'green'}

# Helper function to create scatter plots
def plot_scatter(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df1[x], df1[y], color=colors['Caltech'], s=5, alpha=0.7, label="Caltech")
    ax.scatter(df2[x], df2[y], color=colors['JPL'], s=5, alpha=0.7, label="JPL")
    ax.scatter(df3[x], df3[y], color=colors['Office1'], s=5, alpha=0.7, label="Office 1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig, ax

# Plot 1: Arrival Time vs. Staying Duration
fig1, ax1 = plot_scatter('arrivalMinutes', 'stayingDuration', 'Arrival time (PT)', 'Staying time (minutes)', 'Arrival time against duration of stay')
ax1.set_xticks(range(0, 1441, 120))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
ax1.set_ylim(-75, 2000)

# Plot 2: Charging Duration vs kWh Delivered
fig2, ax2 = plot_scatter('chargingDuration', 'kWhDelivered', 'Charging time (minutes)', 'kWh Delivered', 'Charging duration against energy delivered')
ax2.set_xlim(0, 2000)

# Plot 3: Staying Duration vs kWh Delivered
fig3, ax3 = plot_scatter('stayingDuration', 'kWhDelivered', 'Stay time (minutes)', 'kWh Delivered', 'Durations of stay against energy delivered')
ax3.set_xlim(0, 2000)

# Plot 4: Non-use Duration vs kWh Delivered
fig4, ax4 = plot_scatter('nonuseDuration', 'kWhDelivered', 'Non-use stay time (minutes)', 'kWh Delivered', 'Non-use staying time against energy delivered')
ax4.set_xlim(0, 2000)

# Plot 5: Arrival Time vs kWh Delivered
fig5, ax5 = plot_scatter('arrivalMinutes', 'kWhDelivered', 'Arrival time (PT)', 'kWh Delivered', 'Arrival time against energy delivered')
ax5.set_xticks(range(0, 1441, 120))
ax5.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

# Plot 6: Arrival Date vs kWh Delivered
fig6, ax6 = plot_scatter('arrivalDate', 'kWhDelivered', 'Arrival date', 'kWh Delivered', 'Arrival date against energy delivered')

# Show all plots
plt.show()
