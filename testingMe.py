import ezc3d
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

#set path and read the csv for the POI Data
poi_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/poi_metrics.csv'
poi_metrics = pd.read_csv(poi_path)

#set path and read the csv for the HitTrax Data
hittrax_path = '/Users/leofeingold/Documents/GitHub/openbiomechanics/baseball_hitting/data/poi/hittrax.csv'
hittrax_metrics = pd.read_csv(hittrax_path)

#Merge the data by session swing so it lines up correctly
merged_data = pd.merge(poi_metrics, hittrax_metrics, on='session_swing', how='inner')

#assign varibale x and y for simplicty in regression
x = merged_data['exit_velo_mph_x'].values.reshape(-1, 1)
y = merged_data['la'].values

# Fit a linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values using the model
x_values = np.linspace(x.min(), x.max(), 100).reshape(-1, 1) #creates numpy array for 100 x values between the xMin and xMax and reshapes it into a 1 column by however many rows are neccesary
y_values = model.predict(x_values) #predict the values using the regression


# Add the formula for the line GRAPH: WHY WE NEED BARREL%
plt.text(50,-15, f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}', fontsize=12, color='blue')
plt.scatter(merged_data['exit_velo_mph_x'], merged_data['la'], label='Data')
plt.plot(x_values, y_values, color='red', label='Trendline')
plt.xlabel('Exit Velo')
plt.ylabel('Launch Angle')
plt.title('Exit Velo and Launch Angle: Why we need Barrel%')
plt.legend()
plt.show()

#So now, what factors predict barrel% (we can define as top 8Th percentile EV at some certain LA), what factors predict EV, and are they the same and why?


sns.histplot(merged_data['bat_speed_mph_max_x'], kde=True)
plt.title('Bat Speed Distribution')
plt.xlabel('Bat Speed MPH')
plt.show()

