import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Load data
poi_metrics = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/poi_metrics.csv")
hittrax = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/hittrax.csv")

# Merge datasets
hittrax_poi = poi_metrics.merge(hittrax[['session_swing', 'pitch']], on='session_swing')

# Calculate Driveline Smash Factor
hittrax_poi['smash_factor'] = 1 + ((hittrax_poi['exit_velo_mph_x'] - hittrax_poi['bat_speed_mph_contact_x']) /
                                   (hittrax_poi['pitch'] + hittrax_poi['bat_speed_mph_contact_x']))

# Define mechanical POI columns
mechanical_poi_columns = [
    'bat_torso_angle_connection_x',
    'attack_angle_contact_x',
    'bat_torso_angle_ds_y',
    'hand_speed_blast_bat_mph_max_x',
    'hand_speed_mag_max_x',
    'pelvis_angle_fm_x',
    'pelvis_angle_fm_y',
    'pelvis_angle_fm_z',
    'pelvis_angle_fp_x', 
    'pelvis_angle_fp_y', 
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
    'x_factor_fm_x', 
    'x_factor_fm_y', 
    'x_factor_fm_z',
    'x_factor_fp_x', 
    'x_factor_fp_y', 
    'x_factor_fp_z',
    'pelvis_angular_velocity_fm_x',
    'pelvis_angular_velocity_fp_x',
    'pelvis_angular_velocity_maxhss_x',
    'pelvis_angular_velocity_seq_max_x',
    'pelvis_angular_velocity_stride_max_x',
    'pelvis_angular_velocity_swing_max_x',
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
    'torso_angular_velocity_fm_x',
    'torso_angular_velocity_fp_x',
    'torso_angular_velocity_maxhss_x',
    'torso_angular_velocity_seq_max_x',
    'torso_angular_velocity_stride_max_x',
    'torso_angular_velocity_swing_max_x',
    'upper_arm_speed_mag_fm_x',
    'upper_arm_speed_mag_fp_x',
    'upper_arm_speed_mag_maxhss_x',
    'upper_arm_speed_mag_seq_max_x',
    'upper_arm_speed_mag_stride_max_velo_x',
    'upper_arm_speed_mag_swing_max_velo_x',
    'x_factor_hs_x', 
    'x_factor_hs_y', 
    'x_factor_hs_z',
    'max_cog_velo_x',
]

# Prepare the data
X = hittrax_poi[mechanical_poi_columns]
y = hittrax_poi["smash_factor"]

# Scale values for better regression results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define XGBoost parameters correctly
params = {
    'objective': 'reg:squarederror',  # for regression task
    'max_depth': 5,  # maximum depth of a tree
    'eta': 0.1,  # learning rate
    'subsample': 0.8,  # subsample ratio of the training instances
    'colsample_bytree': 0.8,  # subsample ratio of columns when constructing each tree
    'alpha': 0,  # L1 regularization term
    'lambda': 10,  # L2 regularization term
}

# Number of boosting rounds
num_boost_round = 49

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
r2_scores = []
feature_importance_df = pd.DataFrame()

for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train the model with the corrected parameters
    bst = xgb.train(params, dtrain, num_boost_round)
    
    # Make predictions
    y_pred = bst.predict(dtest)
    
    # Evaluate the model
    mse_scores.append(mean_squared_error(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))
    
    # Determine feature importance
    importance = bst.get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'feature': [mechanical_poi_columns[int(k[1:])] for k in importance.keys()],
        f'importance_{fold}': importance.values()
    }).sort_values(by=f'importance_{fold}', ascending=False)
    
    if feature_importance_df.empty:
        feature_importance_df = importance_df
    else:
        feature_importance_df = feature_importance_df.merge(importance_df, on='feature', how='outer')

# Aggregate feature importance
numeric_cols = feature_importance_df.columns.drop('feature')
feature_importance_df['mean_importance'] = feature_importance_df[numeric_cols].mean(axis=1)

# Display cross-validation results
results_cv = {
    'MSE': np.mean(mse_scores),
    'MAE': np.mean(mae_scores),
    'R2 Score': np.mean(r2_scores),
    'Feature Importance': feature_importance_df[['feature', 'mean_importance']].sort_values(by='mean_importance', ascending=False)
}

print("Cross-Validation Results:")
print(f"MSE: {results_cv['MSE']}")
print(f"MAE: {results_cv['MAE']}")
print(f"R2 Score: {results_cv['R2 Score']}")
print("Feature Importance:")
print(results_cv['Feature Importance'])