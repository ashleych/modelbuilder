#%%
import pandas as pd

import numpy as np

df_macros = pd.read_csv("./input/macro_input.csv")
numerics = ['int16', 'int32', 'int64','float64','float32']
df = df_macros.select_dtypes(include=numerics)
# %%

cols=df.columns.tolist()
cols.sort()
cols

#%%
from statsmodels.tsa.stattools import adfuller,kpss
import pandas as pd


def adfuller_test(series, signif=0.05):
# """
# Perform Augmented Dickey-Fuller to test for Stationarity of the given series
# and print report. Null Hypothesis: Data has unit root and is non-stationary.

# series: time series in pd.Series format
# signif: significance level for P-value to reject Null Hypothesis
# """
    x = adfuller(series, autolag='AIC')

    #using dictionary saves different data types (float, int, boolean)
    output = {'Test Statistic': x[0], 
              'P-value': x[1], 
              'Number of lags': x[2], 
              'Number of observations': x[3],
              f'Reject (signif. level {signif})': x[1] < signif }

    for key, val in x[4].items():
         output[f'Critical value {key}'] = val

    return pd.Series(output)
adf_results=df_macros[cols].apply(lambda x: adfuller_test(x), axis=0) 
adf_results=adf_results.T
# %%
# adf_results.loc(adf_results['Reject (signif. level 0.05)'])
adf_results_passed=adf_results[adf_results['Reject (signif. level 0.05)']==True].index.values
#%%
def kpss_test(timeseries,signif=0.05):
#     KPSS is another test for checking the stationarity of a time series. The null and alternate hypothesis for the KPSS test are opposite that of the ADF test.
# Null Hypothesis: The process is trend stationary.

    x = kpss(timeseries, regression="c", nlags="auto")
    # kpss_output = pd.Series(
    #     kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    # )
    output = {'Test Statistic': x[0], 
              'P-value': x[1], 
              'Number of lags': x[2], 
              f'Accept (signif. level {signif})': x[1] > signif }
    for key, value in x[3].items():
        output["Critical Value (%s)" % key] = value
    return pd.Series(output)
kpss_results=df_macros[cols].apply(lambda x: kpss_test(x), axis=0).T 
kpss_results_passed=kpss_results[kpss_results['Accept (signif. level 0.05)']==True].index.values
kpss_results_passed

#%%
import matplotlib
df_macros['M255'].plot.line()
# %%
stationary_passed=set(kpss_results_passed).intersection(adf_results_passed)
# %%
stationary_passed
# %%
