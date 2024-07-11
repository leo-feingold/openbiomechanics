import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import sklearn

print("Versions:")
print(f"Pandas: {pd.__version__}")
print(f"XGB: {xgb.__version__}")
print(f"Sklearn: {sklearn.__version__}")