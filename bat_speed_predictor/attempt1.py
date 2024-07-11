# import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split #, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# load data
poi_metrics = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/poi_metrics.csv")

'''
Columns:
['session_swing', 'session', 'exit_velo_mph_x', 'blast_bat_speed_mph_x', 'bat_speed_mph_contact_x', 'sweet_spot_velo_mph_contact_x', 
'sweet_spot_velo_mph_contact_y', 'sweet_spot_velo_mph_contact_z', 'bat_torso_angle_connection_x', 'attack_angle_contact_x', 
'bat_speed_mph_max_x', 'bat_speed_xy_max_x', 'bat_torso_angle_ds_x', 'bat_torso_angle_ds_y', 'bat_torso_angle_ds_z', 
'hand_speed_blast_bat_mph_max_x', 'hand_speed_mag_max_x', 'pelvis_angle_fm_x', 'pelvis_angle_fm_y', 'pelvis_angle_fm_z', 
'pelvis_angle_fp_x', 'pelvis_angle_fp_y', 'pelvis_angle_fp_z', 'pelvis_angle_hs_x', 'pelvis_angle_hs_y', 'pelvis_angle_hs_z',
'torso_angle_fm_x', 'torso_angle_fm_y', 'torso_angle_fm_z', 'torso_angle_fp_x', 'torso_angle_fp_y', 'torso_angle_fp_z', 
'torso_angle_hs_x', 'torso_angle_hs_y', 'torso_angle_hs_z', 'upper_arm_speed_mag_max_x', 'x_factor_fm_x', 'x_factor_fm_y', 
'x_factor_fm_z', 'x_factor_fp_x', 'x_factor_fp_y', 'x_factor_fp_z', 'bat_max_x', 'bat_min_x', 'hand_speed_mag_fm_x', 
'hand_speed_mag_fp_x', 'hand_speed_mag_maxhss_x', 'hand_speed_mag_seq_max_x', 'hand_speed_mag_stride_max_velo_x', 
'hand_speed_mag_swing_max_velo_x', 'lead_knee_launchpos_x', 'lead_knee_stride_max_x', 'lead_wrist_fm_x', 'lead_wrist_swing_max_x', 
'pelvis_angular_velocity_fm_x', 'pelvis_angular_velocity_fp_x', 'pelvis_angular_velocity_maxhss_x', 'pelvis_angular_velocity_seq_max_x', 
'pelvis_angular_velocity_stride_max_x', 'pelvis_angular_velocity_swing_max_x', 'pelvis_fm_x', 'pelvis_fm_y', 'pelvis_fm_z', 'pelvis_launchpos_x', 
'pelvis_launchpos_y', 'pelvis_launchpos_z', 'pelvis_loadedpos_x', 'pelvis_load_max_x', 'pelvis_stride_max_x', 'pelvis_stride_max_y', 
'pelvis_stride_max_z', 'pelvis_swing_max_x', 'pelvis_swing_max_y', 'pelvis_swing_max_z', 'rear_elbow_fm_x', 'rear_elbow_fm_z', 
'rear_elbow_launchpos_x', 'rear_elbow_stride_max_x', 'rear_elbow_stride_max_z', 'rear_elbow_swing_max_x', 'rear_elbow_swing_max_z', 
'rear_hip_launchpos_x', 'rear_hip_stride_max_x', 'rear_hip_stride_max_y', 'rear_hip_stride_max_z', 'rear_shoulder_launchpos_x', 
'rear_shoulder_stride_max_x', 'rear_shoulder_stride_max_y', 'rear_shoulder_stride_max_z', 'torso_angular_velocity_fm_x', 
'torso_angular_velocity_fp_x', 'torso_angular_velocity_maxhss_x', 'torso_angular_velocity_stride_max_x', 'torso_angular_velocity_seq_max_x', 
'torso_angular_velocity_swing_max_x', 'torso_fm_x', 'torso_fm_y', 'torso_fm_z', 'torso_launchpos_x', 'torso_launchpos_y', 'torso_launchpos_z', 
'torso_loadedpos_x', 'torso_load_max_x', 'torso_pelvis_fm_x', 'torso_pelvis_launchpos_x', 'torso_pelvis_loadedpos_x', 'torso_pelvis_load_max_x', 
'torso_pelvis_stride_max_x', 'torso_pelvis_stride_max_y', 'torso_pelvis_stride_max_z', 'torso_pelvis_swing_max_x', 'torso_stride_max_x', 
'torso_stride_max_y', 'torso_stride_max_z', 'torso_swing_max_x', 'torso_swing_max_y', 'torso_swing_max_z', 'upper_arm_speed_mag_fm_x', 
'upper_arm_speed_mag_fp_x', 'upper_arm_speed_mag_maxhss_x', 'upper_arm_speed_mag_seq_max_x', 'upper_arm_speed_mag_stride_max_velo_x', 
'upper_arm_speed_mag_swing_max_velo_x', 'x_factor_hs_x', 'x_factor_hs_y', 'x_factor_hs_z', 'max_cog_velo_x']
'''

poi_columns = [
    'torso_angular_velocity_fm_x', 
    'torso_angular_velocity_fp_x', 
    'torso_angular_velocity_maxhss_x', 
    'torso_angular_velocity_stride_max_x', 
    'torso_angular_velocity_seq_max_x', 
    'torso_angular_velocity_swing_max_x', 
    'x_factor_fm_x',
    'x_factor_fm_y', 
    'x_factor_fm_z', 
    'x_factor_fp_x', 
    'x_factor_fp_y', 
    'x_factor_fp_z',
    'bat_torso_angle_ds_x', 
    'bat_torso_angle_ds_y', 
    'bat_torso_angle_ds_z', 
    'pelvis_angle_fm_x', 
    'pelvis_angle_fm_y', 
    'pelvis_angle_fm_z', 
    'pelvis_angle_fp_x', 
    'pelvis_angle_fp_y', 
    'blast_bat_speed_mph_x',
    'bat_torso_angle_connection_x', 
    'attack_angle_contact_x', 
    'pelvis_angle_fp_z', 
    'pelvis_angle_hs_x', 
    'pelvis_angle_hs_y', 
    'pelvis_angle_hs_z',
    'torso_angle_fm_x', 
    'torso_angle_fm_y', 
    'torso_angle_fm_z', 
    'torso_angle_fp_x', 
    'torso_angle_fp_y', 
    'torso_angle_fp_z', 
    'torso_angle_hs_x', 
    'torso_angle_hs_y', 
    'torso_angle_hs_z', 
    'upper_arm_speed_mag_max_x', 
    'lead_knee_launchpos_x', 
    'lead_knee_stride_max_x', 
    'lead_wrist_fm_x', 
    'lead_wrist_swing_max_x', 
    'pelvis_angular_velocity_fm_x',
    'pelvis_angular_velocity_fp_x', 
    'pelvis_angular_velocity_maxhss_x', 
    'pelvis_angular_velocity_seq_max_x', 
    'pelvis_angular_velocity_stride_max_x', 
    'pelvis_angular_velocity_swing_max_x', 
    'pelvis_fm_x', 
    'pelvis_fm_y', 
    'pelvis_fm_z', 
    'pelvis_launchpos_x', 
    'pelvis_launchpos_y', 
    'pelvis_launchpos_z', 
    'pelvis_loadedpos_x', 
    'pelvis_load_max_x', 
    'pelvis_stride_max_x', 
    'pelvis_stride_max_y', 
    'pelvis_stride_max_z', 
    'pelvis_swing_max_x', 
    'pelvis_swing_max_y', 
    'pelvis_swing_max_z', 
    'rear_elbow_fm_x', 
    'rear_elbow_fm_z', 
    'rear_elbow_launchpos_x', 
    'rear_elbow_stride_max_x', 
    'rear_elbow_stride_max_z', 
    'rear_elbow_swing_max_x', 
    'rear_elbow_swing_max_z', 
    'rear_hip_launchpos_x', 
    'rear_hip_stride_max_x', 
    'rear_hip_stride_max_y', 
    'rear_hip_stride_max_z', 
    'rear_shoulder_launchpos_x', 
    'rear_shoulder_stride_max_x', 
    'rear_shoulder_stride_max_y', 
    'rear_shoulder_stride_max_z', 
    'torso_fm_x', 
    'torso_fm_y', 
    'torso_fm_z', 
    'torso_launchpos_x', 
    'torso_launchpos_y', 
    'torso_launchpos_z', 
    'torso_loadedpos_x', 
    'torso_load_max_x', 
    'torso_pelvis_fm_x', 
    'torso_pelvis_launchpos_x', 
    'torso_pelvis_loadedpos_x', 
    'torso_pelvis_load_max_x', 
    'torso_pelvis_stride_max_x', 
    'torso_pelvis_stride_max_y', 
    'torso_pelvis_stride_max_z', 
    'torso_pelvis_swing_max_x', 
    'torso_stride_max_x', 
    'torso_stride_max_y', 
    'torso_stride_max_z', 
    'torso_swing_max_x', 
    'torso_swing_max_y', 
    'torso_swing_max_z', 
    'upper_arm_speed_mag_fm_x', 
    'upper_arm_speed_mag_fp_x', 
    'upper_arm_speed_mag_maxhss_x', 
    'upper_arm_speed_mag_seq_max_x', 
    'upper_arm_speed_mag_stride_max_velo_x', 
    'upper_arm_speed_mag_swing_max_velo_x', 
    'x_factor_hs_x', 
    'x_factor_hs_y', 
    'x_factor_hs_z', 
    'max_cog_velo_x'

]


poi_metrics = poi_metrics[poi_columns].dropna()
print(f"Size After Drops: {poi_metrics.shape}")

# prepare the data
X = poi_metrics.drop('blast_bat_speed_mph_x', axis=1)
y = poi_metrics["blast_bat_speed_mph_x"]

# scale values for better regression results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# define XGBoost parameters correctly
params = {
    'objective': 'reg:squarederror',  # for regression task
    'max_depth': 5,  # maximum depth of a tree
    'eta': 0.1,  # learning rate
    'subsample': 0.8,  # subsample ratio of the training instances
    'colsample_bytree': 0.8,  # subsample ratio of columns when constructing each tree
    'alpha': 0,  # L1 regularization term
    'lambda': 10,  # L2 regularization term
}
num_boost_round = 50


# split into training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
    
# train the model with the specified parameters

# can I use this instead:
#simple_model = xgb.XGBRegressor()
#simple_model.fit(X_train, y_train)


bst = xgb.train(params, dtrain, num_boost_round)
    
# make predictions
y_pred = bst.predict(dtest)

# evaluate model
mse_score = (mean_squared_error(y_test, y_pred))
mae_score = (mean_absolute_error(y_test, y_pred))
r2score = (r2_score(y_test, y_pred))
    
# determine feature importance
importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': [X.columns[int(k[1:])] for k in importance.keys()],
    'importance': importance.values()
}).sort_values(by='importance', ascending=False)

print(f"MSE: {mse_score}")
print(f"MAE: {mae_score}")
print(f"r^2: {r2score}")

print(importance_df)