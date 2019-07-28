import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly
from fbprophet import Prophet

if __name__ == "__main__":
    # execute only if run as a script
    logging.info("Load Files...")
    logging.info("...2012")
    df2012 = pd.read_csv("data_local/entreprises-immatriculees-2012.csv", sep=";", dtype="str")
    df2012 = df2012[["Date d'immatriculation"]]
    df2012 = pd.to_datetime(df2012["Date d'immatriculation"])
    logging.info("...2013")
    df2013 = pd.read_csv("data_local/entreprises-immatriculees-2013.csv", sep=";", dtype="str")
    df2013 = df2013[["Date d'immatriculation"]]
    df2013 = pd.to_datetime(df2013["Date d'immatriculation"])
    logging.info("...2014")
    df2014 = pd.read_csv("data_local/entreprises-immatriculees-2014.csv", sep=";", dtype="str")
    df2014 = df2014[["Date d'immatriculation"]]
    df2014 = pd.to_datetime(df2014["Date d'immatriculation"])
    logging.info("...2015")
    df2015 = pd.read_csv("data_local/entreprises-immatriculees-2015.csv", sep=";", dtype="str")
    df2015 = df2015[["Date immatriculation"]]
    df2015 = pd.to_datetime(df2015["Date immatriculation"])
    logging.info("...2016")
    df2016 = pd.read_csv("data_local/entreprises-immatriculees-2016.csv", sep=";", dtype="str")
    df2016 = df2016[["Date immatriculation"]]
    df2016 = pd.to_datetime(df2016["Date immatriculation"])
    logging.info("...2017")
    df2017 = pd.read_csv("data_local/entreprises-immatriculees-2017.csv", sep=";", dtype="str")
    df2017 = df2017[["Date immatriculation"]]
    df2017 = pd.to_datetime(df2017["Date immatriculation"])
    logging.info("...concat all files")
    df = pd.concat([df2012, df2013, df2014, df2015, df2016, df2017])
    df = pd.DataFrame(df)
    df["y"] = 1
    df.columns = ["ds", "y"]
    df = df.groupby(by=["ds"], as_index=False).count()
    logging.info("... preview dataframe")
    print(df.head(5))

    logging.info("Plot Initial ...")
    plt.figure()
    plt.plot_date(df["ds"], df["y"])
    plt.show()

    logging.info("Forecasting...")
    m = Prophet()
    m.fit(df)
    logging.info("...Make future dataframe")
    future = m.make_future_dataframe(periods=365)
    logging.info("...Preview future df")
    print(future.tail())
    logging.info("...Predict")
    forecast = m.predict(future)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    logging.info("Plot results")
    logging.info("...Figure 1 Forecast")
    fig1 = m.plot(forecast)
    logging.info("...Figure 2 components")
    fig2 = m.plot_components(forecast)
