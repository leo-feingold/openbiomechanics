import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

# set random seed for reproducibility
np.random.seed(42)

# load data
poi_metrics = pd.read_csv("/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/poi_metrics.csv")

poi_columns = [
    'blast_bat_speed_mph_x',
    'bat_torso_angle_connection_x', 
    'attack_angle_contact_x', 
    'bat_torso_angle_ds_x', 
    'bat_torso_angle_ds_y', 
    'bat_torso_angle_ds_z', 
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
    'torso_angular_velocity_fm_x', 
    'torso_angular_velocity_fp_x', 
    'torso_angular_velocity_maxhss_x', 
    'torso_angular_velocity_stride_max_x', 
    'torso_angular_velocity_seq_max_x', 
    'torso_angular_velocity_swing_max_x', 
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

# handle na values
poi_metrics = poi_metrics[poi_columns].dropna()
print(poi_metrics.shape)

# prepare the data
X = poi_metrics.drop('blast_bat_speed_mph_x', axis=1)
X = X.apply(pd.to_numeric, errors='coerce')
cols_with_na = X.columns[X.isna().any()].tolist()
print(f"Columns with NA values: {cols_with_na}")
X = X.dropna(axis=1, how='any')
print(f"X Shape: {X.shape}")
y = poi_metrics["blast_bat_speed_mph_x"]

# Check for multicollinearity and remove highly correlated features
threshold = 5.0  # Variance Inflation Factor threshold
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
X = X.loc[:, vif_data[vif_data["VIF"] < threshold]["feature"]]

# scale values for better regression results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# initialize XGBRegressor
xgb_model = xgb.XGBRegressor(random_state=42)

# perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# train the best model
best_model = grid_search.best_estimator_

# make predictions
y_pred = best_model.predict(X_test)

# evaluate model
mse_score = mean_squared_error(y_test, y_pred)
mae_score = mean_absolute_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)

print(f"MSE: {mse_score}")
print(f"MAE: {mae_score}")
print(f"r^2: {r2score}")

# determine feature importance using SHAP
explainer = shap.Explainer(best_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=X.columns)

# plot data for important features using SHAP
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar")

#Best parameters: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.7}
'''
MSE: 18.07617622171404
MAE: 3.0661744807316706
r^2: 0.45060358294465774
'''