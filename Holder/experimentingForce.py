import ezc3d
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

#create dataframes for each csv
force_file_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/full_sig/force_plate.csv'
data = pd.read_csv(force_file_path)

poi_file_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/poi_metrics.csv'
poi_data = pd.read_csv(poi_file_path)

hittrax_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/hittrax.csv'
hittrax_data = pd.read_csv(hittrax_path)

metadata_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/metadata.csv'
metadata_csv = pd.read_csv(metadata_path)

# Filter rows where time and contact_time are equal, essentially only looking at what the force looks like at contact
contact_indices = data[data['time'] == data['contact_time']].index

# Print the contact indices
#print("Contact indices:", contact_indices)

# Filter data using the contact indices
filtered_data = data.loc[contact_indices]

# Save the filtered data to a new CSV
filtered_data.to_csv("contact_time_data.csv", index=False)

#merge the filtered data and the poi and hittrax data
merged_data_1 = pd.merge(poi_data, filtered_data, on='session_swing', how='inner')
merged_data_2 = pd.merge(merged_data_1, metadata_csv, on='session_swing', how='inner')
merged_data = pd.merge(merged_data_2, hittrax_data, on='session_swing', how='inner')

#choose columns to assess:
interesting_columns = ['lead_force_x','lead_force_y','lead_force_z','rear_force_x','rear_force_y','rear_force_z']
col = 'hand_speed_blast_bat_mph_max_x'

#assess correlations of attack angle and forces
for j in range(len(interesting_columns)):
    correlation = merged_data.corr(numeric_only=True)[col][interesting_columns[j]]
    print("Correlation of", col, "and" , interesting_columns[j] , "is" , correlation)

'''
correlations = merged_data.corr(numeric_only=True)['rear_force_z']
correlations = abs(correlations)
print(correlations.sort_values(ascending=False).head(20))
'''

#print(merged_data['lead_force_y'][100], merged_data['hitter_side'][100])

myVal = merged_data.corr(numeric_only=True)['bat_speed_mph_contact_x']['exit_velo_mph_x_x'] #UMM WHAT THE FUCK HAPPENED HERE

#print("Correlation of EV and Bat Speed:" , myVal)