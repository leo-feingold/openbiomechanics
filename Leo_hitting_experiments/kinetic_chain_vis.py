import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''

['session_swing', 'time', 'lead_elbow_angular_velocity_x', 'lead_elbow_angular_velocity_y', 
'lead_elbow_angular_velocity_z', 'lead_hand_global_angular_velocity_x', 'lead_hand_global_angular_velocity_y', 
'lead_hand_global_angular_velocity_z', 'lead_hip_angular_velocity_x', 'lead_hip_angular_velocity_y', 
'lead_hip_angular_velocity_z', 'lead_knee_angular_velocity_x', 'lead_knee_angular_velocity_y', 
'lead_knee_angular_velocity_z', 'lead_shoulder_angular_velocity_x', 'lead_shoulder_angular_velocity_y', 
'lead_shoulder_angular_velocity_z', 'lead_shoulder_global_angular_velocity_x', 'lead_shoulder_global_angular_velocity_y', 
'lead_shoulder_global_angular_velocity_z', 'lead_wrist_angular_velocity_x', 'lead_wrist_angular_velocity_y', 
'lead_wrist_angular_velocity_z', 'pelvis_angular_velocity_x', 'pelvis_angular_velocity_y', 'pelvis_angular_velocity_z', 
'rear_elbow_angular_velocity_x', 'rear_elbow_angular_velocity_y', 'rear_elbow_angular_velocity_z', 
'rear_hand_global_angular_velocity_x', 'rear_hand_global_angular_velocity_y', 'rear_hand_global_angular_velocity_z', 
'rear_hip_angular_velocity_x', 'rear_hip_angular_velocity_y', 'rear_hip_angular_velocity_z', 
'rear_knee_angular_velocity_x', 'rear_knee_angular_velocity_y', 'rear_knee_angular_velocity_z',
'rear_shoulder_angular_velocity_x', 'rear_shoulder_angular_velocity_y', 'rear_shoulder_angular_velocity_z', 
'rear_shoulder_global_angular_velocity_x', 'rear_shoulder_global_angular_velocity_y', 
'rear_shoulder_global_angular_velocity_z', 'rear_wrist_angular_velocity_x', 'rear_wrist_angular_velocity_y', 
'rear_wrist_angular_velocity_z', 'torso_angular_velocity_x', 'torso_angular_velocity_y', 
'torso_angular_velocity_z', 'torso_pelvis_angular_velocity_x', 'torso_pelvis_angular_velocity_y', 
'torso_pelvis_angular_velocity_z', 'fp_10_time', 'fp_100_time', 'contact_time']

'''

swing_session = "47_2"

landmark_data = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/full_sig/joint_velos.csv")
landmark_data = landmark_data[landmark_data["session_swing"] == swing_session]

force_data = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/full_sig/force_plate.csv")
force_data = force_data[force_data["session_swing"] == swing_session]

# Create a figure and a set of subplots
fig, ax1 = plt.subplots()

# Plot landmark data on the first y-axis
ax1.plot(landmark_data.time, landmark_data.pelvis_angular_velocity_x, 'b-', label='Pelvis Angular Velocity (X)')
ax1.plot(landmark_data.time, landmark_data.torso_angular_velocity_x, color='red', label='Torso Angular Velocity (X)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Pelvis and Torso Angular Velocity (X)', color='black')
ax1.tick_params('y', colors='black')

# Add vertical lines for fp_100_time and contact_time
fp_100_time = landmark_data["fp_100_time"].iloc[0]
contact_time = landmark_data["contact_time"].iloc[0]
max_pelvis_velo_idx = landmark_data["pelvis_angular_velocity_x"].argmax()
max_pelvis_time = landmark_data.time.iloc[max_pelvis_velo_idx]

min_point_y = min(min(landmark_data.pelvis_angular_velocity_x.dropna()), min(landmark_data.torso_angular_velocity_x.dropna()))
print(min_point_y)


# Get the y-axis limits
ax1.set_ylim(-100, 300)  # Adjust these values as needed
ymin, ymax = ax1.get_ylim()
print(ymin)

# Plot markers at the bottom of the graph
ax1.plot(fp_100_time, -100, '|', color='black', markersize=20, label="Foot Plant, Contact")  # Black marker for fp_100_time
ax1.plot(contact_time, -100, '|', color='black', markersize=20)  # Black marker for contact_time


# Add a title and legend
plt.title(f'Pelvis and Torso Angular Velocity (Swing Session: {swing_session})')
#fig.tight_layout()
ax1.legend(loc='upper left')

# Show the plot
plt.show()
