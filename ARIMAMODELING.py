import pandas as pd 
import warnings 
import itertools
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Defaults 
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')


# Load dataset
data = pd.read_csv("international-airline-passengers.csv", engine="python", skipfooter=3)

# Pre-processing 
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
data.set_index(['Month'], inplace=True)

# Plot the data
data.plot()
plt.ylabel("Monthly airline passengers (x1000)")
plt.xlabel("Date")
plt.show()



# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p paramters to take any value between 0 and 3
p = range(0, 4)

# Generate all different combinations of p, q, and q triplets
pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print("SARIMAX: {} x {}".format(pdq[1], seasonal_pdq[1]))
print("SARIMAX: {} x {}".format(pdq[1], seasonal_pdq[2]))
print("SARIMAX: {} x {}".format(pdq[2], seasonal_pdq[3]))
print("SARIMAX: {} x {}".format(pdq[2], seasonal_pdq[4]))


train_data = data['1949-01-01':'1959-12-01']
test_data = data['1960-01-01':'1960-12-01']


warnings.filterwarnings("ignore") # Specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            ORDER=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationary=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            print("SARIMAX{}x{} - AIC:{}".format(param, param_seasonal, results.aic, end="\r"))
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue


print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))

# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)


results = mod.fit()

results.plot_diagnostics(figsize=(20, 14))
plt.show()