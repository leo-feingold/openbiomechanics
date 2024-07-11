import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/full_sig/landmarks.csv")

# Group by session_swing
grouped_df = df.groupby("session_swing")

# Create a new column 'redo_time'
df = df.assign(
    redo_time = lambda x: df.contact_time - df.time
)

# Filter the DataFrame
swing = '103_5'
maskSwing = f"session_swing == '{swing}'"
masked_df = df.query(maskSwing)

# Plot the graph
plt.plot(masked_df.redo_time, masked_df.centerofmass_x)
plt.xlabel('Time (t=0 is contact)')
plt.ylabel('Center of Mass X')
plt.title('Center of Mass X vs Time')
plt.suptitle(f"Swing: {swing}")
plt.show()