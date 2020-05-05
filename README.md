# Time Series
Time series is a collection of data points indexed in time order (stock prices, temperatures, weekly sales …). In this article, we will be using an open dataset, which records daily the registration of a company on the French company register.

Time series forecast forecasting is building a model based on the observed value of the same variable to predict future values of that same variable. This is different from regression approaches where the prediction of future values is based on the observation of independent features (i.e. any feature but the variable we are trying to predict).

# Prophet
In September 2017, Facebook published a paper called "Forecasting at scale": https://peerj.com/preprints/3190/ resulting in the creation of Prophet. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

# Usage

```shell
python3 forecast.py 
```
