# load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from statsmodels.tsa.seasonal import seasonal_decompose,STL
from statsmodels.tsa.stattools import adfuller
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

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
plt.xticks(np.arange(6,203,12),np.arange(1992,2009,1))
fig.autofmt_xdate()
plt.tight_layout()

# Step b:Exploration
decomposition = STL(df.y,period=12).fit()
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(10,8))
ax1.plot(decomposition.observed)
ax1.set_ylabel("Observed")

ax2.plot(decomposition.trend)
ax2.set_ylabel("Trend")

ax3.plot(decomposition.seasonal)
ax3.set_ylabel("Seasonal")

ax4.plot(decomposition.resid)
ax4.set_ylabel("Residuals")

plt.xticks(np.arange(6,203,12),np.arange(1992,2009,1))
fig.autofmt_xdate()
plt.tight_layout()

# step c: 
## the SARIMA() model is picked because there is seasonality 
## we dont pick the VAR because we not dealing with multiple interrelated timeseries data 

# Step d .1 ,Check for stationarity 
ad_fuller_result = adfuller(df.y)
print(f"ADF statistics : {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

# difference 1st order 
y_diff = np.diff(df.y,n=1)
ad_fuller_result=adfuller(y_diff)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# differencing 1st order with n=12
y_diff_seasonal = np.diff(df.y,n=12)
ad_fuller_result=adfuller(y_diff_seasonal)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}') 

#Therefore d = 1,D=1,m=12 -> SARIMA(p,1,q)(P,1,Q)_12 

# model selection
# we chose the parameters that minimize the Akaike information criterion

train = df.y[:168]
test= df.y[168:]

# function to find values of the parameters 
from typing import Union
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX



# residual testing
# use plot_diagnostics to see if the residuals behave like white noise
SARIMA_model = SARIMAX(train,order=(3,1,1),seasonal_order=(1,1,3,12),simple_differencing=False)

SARIMA_model_fit = SARIMA_model.fit(disp=False)
SARIMA_model_fit.plot_diagnostics(figsize=(10,8))


# Ljung-Box test
# This determines whether the residuals are independent and uncorrelated
# So we need p-value>0.05
residuals = SARIMA_model_fit.resid
lb_test_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
pvalues = lb_test_results["lb_pvalue"]
print("Ljung-Box test p-values:", pvalues)

# forecasting 

def rolling_forecast(df:pd.DataFrame,train_len:int,horizon:int,window:int,method:str)-> list:
    """A function to generate predictions over the entire test set with a window of 12 months"""
    total_len = train_len + horizon
    end_idx = train_len

    if method == 'last_season':
        pred_last_season = []
        for i in range(train_len,total_len,window):
            last_season = df['y'][i-window:i].values
            pred_last_season.extend(last_season)
        return pred_last_season
    elif method == 'SARIMA':
        pred_SARIMA = []
        for i in range(train_len,total_len,window):
            model = SARIMAX(df["y"][:i],order=(3,1,1),seasonal_order=(1,1,3,12),simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_predictions(0,i+window-1)
            oos_pred = predictions.predicted_mean.iloc[-window]
            pred_SARIMA.extend(oos_pred)
        return pred_SARIMA
    
# define a few parameters

pred_df = df[168:]

TRAIN_LEN = 168
HORIZON = 36
WINDOW = 12

# perform the prediction from the SARIMA model

pred_df['SARIMA'] = rolling_forecast(df,TRAIN_LEN,HORIZON,WINDOW)

