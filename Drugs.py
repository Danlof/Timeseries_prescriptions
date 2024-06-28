# load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# import data 
df = pd.read_csv("/media/danlof/dan_files/data_science_codes/Timeseries/Antidiabetic/AusAntidiabeticDrug.csv")
df.head()

df.tail()

df.shape

# Visualization
fig, ax = plt.subplots()
ax.plot(df.y)
ax.set_xlabel('Date')
ax.set_ylabel('Number of drug prescriptions')
plt.xticks(np.)
