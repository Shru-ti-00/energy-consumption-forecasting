import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split,cross_val_score)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import(mean_absolute_error, mean_squared_error, r2_score)

from xgboost import XGBRegressor

import joblib

# Load Dataset
df=pd.read_csv("powerconsumption.csv.zip")

#------DATA CLEANING-------
# Converting Datetime

df['Datetime']=pd.to_datetime(df['Datetime'])


#---------------FEATURE ENGINEERING------------------

# Time Features
df['Hour']=df['Datetime'].dt.hour

df['Day']=df['Datetime'].dt.day

df['Month']=df['Datetime'].dt.month

df['DayOfWeek'] = df['Datetime'].dt.dayofweek

df['DayOfYear']=df['Datetime'].dt.dayofyear

df['WeekOfYear']=(
    df['Datetime']
    .dt
    .isocalendar()
    .week
    .astype(int)
)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Lag Features
df['Lag_1'] = (
    df['PowerConsumption_Zone1']
    .shift(1)
)

df['Lag_24'] = (
    df['PowerConsumption_Zone1']
    .shift(24)
)

#------------------------------ROLLING FEATURES---------------------------------
df['Rolling_Mean_3'] = (
    df['PowerConsumption_Zone1']
    .rolling(window=3)
    .mean()
)

df['Rolling_Mean_24'] = (
    df['PowerConsumption_Zone1']
    .rolling(window=24)
    .mean()
)

#--------------------INTERACTION FEATURES---------------------

df['Temp_Humidity'] = (
    df['Temperature']
    * df['Humidity']
)

df['Temp_Wind'] = (
    df['Temperature']
    * df['WindSpeed']
)


#-----------------REMOVE MISSING VALUES--------------------

df.dropna(inplace=True)

# -------------REMOVE OUTLIERS-------------

Q1 = df['PowerConsumption_Zone1'].quantile(0.25)

Q3 = df['PowerConsumption_Zone1'].quantile(0.75)

IQR = Q3 - Q1

df = df[
    ~(
        (df['PowerConsumption_Zone1'] < (Q1 - 1.5 * IQR)) |
        (df['PowerConsumption_Zone1'] > (Q3 + 1.5 * IQR))
    )
]

#--------FEATURES AND TARGET---------

X=df.drop([
    'Datetime',
    'PowerConsumption_Zone1',
    'PowerConsumption_Zone2',
    'PowerConsumption_Zone3'], axis=1)

y=df['PowerConsumption_Zone1']



#------------TRAIN TEST SPLIT-------------

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

#-----------------BUILD XGBOOST MODEL---------------

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)


#---------------------TRAIN MODEL---------------------

model.fit(X_train,y_train)

print("\n✅ MOdel Training Completed")

#---------------------PREDICTIONS----------------------

y_pred=model.predict(X_test)

#-------------------EVALUATION METRICS---------------------

mae=mean_absolute_error(y_test,y_pred)

mse=mean_squared_error(y_test,y_pred)

rmse=np.sqrt(mse)

r2=r2_score(y_test,y_pred)

accuracy=r2 * 100

print("\n📊 Model Evaluations")
print(f"MAE       : {mae:.2f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R2 Score  : {r2:.4f}")
print(f"Accuracy  : {accuracy:.2f}%")

scores=cross_val_score(model,X,y,cv=5,scoring='r2')

print("\n📈 Cross Validation scores")

print(scores)

print(f"\nAverage CV score : {scores.mean():.4f}")

#----------------------------ACTUAL Vs PREDICTED GRAPH-----------------------------

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:200],label='Actual')
plt.plot(y_pred[:200],label='Predicted')

plt.title("Actual vs Predicted Energy Consumption")

plt.xlabel("Samples")
plt.ylabel("Power Consumption")

plt.legend()

plt.show()

#----------------------FEATURE IMPORTANCE-------------------------

importance=model.feature_importances_

feature_names=X.columns

feature_df=pd.DataFrame({'Features': feature_names,'Importance': importance})

feature_df=feature_df.sort_values(by='Importance',ascending=False)

plt.figure(figsize=(12,8))

sns.barplot(x='Importance',y='Features',data=feature_df)

plt.title("Feature Importance")

plt.show()

#-------------------SAVE MODEL-------------------

joblib.dump(model,"xgboost_energy_model.pkl")

print("\n✅ Model Saved Succesfully")