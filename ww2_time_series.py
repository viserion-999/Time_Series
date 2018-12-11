import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
import plotly.plotly as py # visualization library
py.sign_in(username='Anagha99', api_key='m8COGJDPN0gLnaCzoRdB')

import plotly.graph_objs as go # plotly graphical object
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
# init_notebook_mode(connected=True)
import os
print(os.listdir("../ww2_data"))
import warnings
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.
# Any results you write to the current directory are saved as output.

#1. Loading the data

# bombing data
aerial = pd.read_csv("../ww2_data/operations.csv")

# first weather data that includes locations like country, latitude and longitude.
weather_station_location = pd.read_csv("../ww2_data/Weather Station Locations.csv")

# Second weather data that includes measured min, max and mean temperatures
weather = pd.read_csv("../ww2_data/Summary of Weather.csv")


#2. Cleaning Data
# drop countries that are NaN
# drop countries that are NaN
aerial = aerial[pd.isna(aerial.Country)==False]
# drop if target longitude is NaN
aerial = aerial[pd.isna(aerial['Target Longitude'])==False]
# Drop if takeoff longitude is NaN
aerial = aerial[pd.isna(aerial['Takeoff Longitude'])==False]
# drop unused features

drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',
             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',
             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',
             'High Explosives', 'High Explosives Type','Mission Type',
             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',
             'Incendiary Devices', 'Incendiary Devices Type',
             'Incendiary Devices Weight (Pounds)',
             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',
             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',
             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',
             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']
aerial.drop(drop_list, axis=1,inplace = True)

#dropping off outliers
aerial = aerial[ aerial.iloc[:,8]!="4248"] # drop this takeoff latitude
aerial = aerial[ aerial.iloc[:,9]!=1355]   # drop this takeoff longitude

print(aerial.info())

#cleaing weather station data
# what we will use only
weather_station_location = weather_station_location.loc[:,["WBAN","NAME","STATE/COUNTRY ID","Latitude","Longitude"] ]
weather_station_location.info()

#cleaning weather data
# what we will use only
weather = weather.loc[:,["STA","Date","MeanTemp"] ]
weather.info()


#Initial DataVisualisation

#Plot1. How many countries are there in the list
#print(aerial['Country'])
current_palette = 'colorblind'
current_palette = sns.color_palette()

# plt.figure(figsize=(12,8))
# sns.countplot(aerial['Country'])
#
# #plt.title("Countries")
# #plt.savefig('countries.png')
# #plt.show()

# Plot 2: Top target countries
# print(aerial['Target Country'].value_counts()[:10])
# plt.figure(figsize=(12,8))
# sns.countplot(aerial['Target Country'])
# plt.xticks(rotation=90)
#
# plt.title("Countries which are target of attacks")
# plt.savefig('target_countries.png')
# plt.show()

#Plot 3. Most used aircraft
# data = aerial['Aircraft Series'].value_counts()
# print(data[:10])
# data = [go.Bar(
#             x=data[:10].index,
#             y=data[:10].values,
#             hoverinfo = 'text',
#             marker = dict(color = 'rgba(177, 14, 22, 0.5)',
#                              line=dict(color='rgb(0,0,0)',width=1.5)),
#     )]
#
# layout = dict(
#     title = 'Aircraft Series',
# )
# fig = go.Figure(data=data, layout=layout)
#
# py.plot(fig)

##Plot 4 :Attack bases

# aerial["color"] = ""
# aerial.color[aerial.Country == "USA"] = "rgb(0,116,217)"
# aerial.color[aerial.Country == "GREAT BRITAIN"] = "rgb(255,65,54)"
# aerial.color[aerial.Country == "NEW ZEALAND"] = "rgb(133,20,75)"
# aerial.color[aerial.Country == "SOUTH AFRICA"] = "rgb(255,133,27)"
#
# data = [dict(
#     type='scattergeo',
#     lon = aerial['Takeoff Longitude'],
#     lat = aerial['Takeoff Latitude'],
#     hoverinfo = 'text',
#     text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
#     mode = 'markers',
#     marker=dict(
#         sizemode = 'area',
#         sizeref = 1,
#         size= 10 ,
#         line = dict(width=1,color = "white"),
#         color = aerial["color"],
#         opacity = 0.7),
# )]
#
# layout = dict(
#     title = 'Countries Take Off Bases ',
#     hovermode='closest',
#     geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
#                countrywidth=1, projection=dict(type='mercator'),
#               landcolor = 'rgb(217, 217, 217)',
#               subunitwidth=1,
#               showlakes = True,
#               lakecolor = 'rgb(255, 255, 255)',
#               countrycolor="rgb(5, 5, 5)")
# )
# fig = go.Figure(data=data, layout=layout)
# py.plot(fig)
#Resut: https://plot.ly/~Anagha99/6/countries-take-off-bases/#/

#Plot5: Bombing Paths
#
# airports = [dict(
#     type='scattergeo',
#     lon=aerial['Takeoff Longitude'],
#     lat=aerial['Takeoff Latitude'],
#     hoverinfo='text',
#     text="Country: " + aerial.Country + " Takeoff Location: " + aerial["Takeoff Location"] + " Takeoff Base: " + aerial[
#         'Takeoff Base'],
#     mode='markers',
#     marker=dict(
#         size=5,
#         #color=aerial["color"],
#         line=dict(
#             width=1,
#             color="white"
#         )
#     ))]
#
# targets = [dict(
#     type='scattergeo',
#     lon=aerial['Target Longitude'],
#     lat=aerial['Target Latitude'],
#     hoverinfo='text',
#     text="Target Country: " + aerial["Target Country"] + " Target City: " + aerial["Target City"],
#     mode='markers',
#     marker=dict(
#         size=1,
#         color="red",
#         line=dict(
#             width=0.5,
#             color="red"
#         )
#     ))]
#
# flight_paths = []
# for i in range(len(aerial['Target Longitude'])):
#     flight_paths.append(
#         dict(
#             type='scattergeo',
#             lon=[aerial.iloc[i, 9], aerial.iloc[i, 16]],
#             lat=[aerial.iloc[i, 8], aerial.iloc[i, 15]],
#             mode='lines',
#             line=dict(
#                 width=0.7,
#                 color='black',
#             ),
#             opacity=0.6,
#         )
#     )
#
# layout = dict(
#     title='Bombing Paths from Attacker Country to Target ',
#     hovermode='closest',
#     geo=dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
#              countrywidth=1, projection=dict(type='mercator'),
#              landcolor='rgb(217, 217, 217)',
#              subunitwidth=1,
#              showlakes=True,
#              lakecolor='rgb(255, 255, 255)',
#              countrycolor="rgb(5, 5, 5)")
# )
#
# fig = dict(data=flight_paths + airports + targets, layout=layout)
#
# py.plot(fig)
#plot: https://plot.ly/~Anagha99/8/bombing-paths-from-attacker-country-to-target/#/


# #Theater of Operations
# print(aerial['Theater of Operations'].value_counts())
# plt.figure(figsize=(22,10))
# sns.countplot(aerial['Theater of Operations'])
# plt.show()

# * Lets focus **USA and BURMA war**
# * In this war USA bomb BURMA( KATHA city) from 1942 to 1945.
# * The closest weather station to this war is **BINDUKURI** and it has temperature record from 1943 to 1945.
# * Now lets visualize this situation. But before visualization, we need to make date features date time object.

#taking only bindukuri station data
weather_station_id = weather_station_location[weather_station_location.NAME == "BINDUKURI"].WBAN
weather_bin = weather[weather.STA == 32907]
weather_bin["Date"] = pd.to_datetime(weather_bin["Date"])
# plt.figure(figsize=(12,8))
# plt.plot(weather_bin.Date,weather_bin.MeanTemp)
# plt.title("Mean Temperature of Bindukuri Area")
# plt.xlabel("Date")
# plt.ylabel("Mean Temperature")
# plt.show()

#
# As you can see, we have temperature measurement from 1943 to 1945.
# * Temperature ossilates between 12 and 32 degrees.
# * Temperature of  winter months is colder than  temperature of  summer months.

#Getting attack data only for our area of work: Bindukuri
aerial = pd.read_csv("../ww2_data/operations.csv")
aerial["year"] = [ each.split("/")[2] for each in aerial["Mission Date"]]
aerial["month"] = [ each.split("/")[0] for each in aerial["Mission Date"]]
aerial = aerial[aerial["year"]>="1943"]
aerial = aerial[aerial["month"]>="8"]

aerial["Mission Date"] = pd.to_datetime(aerial["Mission Date"])

attack = "USA"
target = "BURMA"
city = "KATHA"

aerial_war = aerial[aerial.Country == attack]
aerial_war = aerial_war[aerial_war["Target Country"] == target]
aerial_war = aerial_war[aerial_war["Target City"] == city]

liste = []
aa = []
for each in aerial_war["Mission Date"]:
    dummy = weather_bin[weather_bin.Date == each]
    liste.append(dummy["MeanTemp"].values)
aerial_war["dene"] = liste
for each in aerial_war.dene.values:
    aa.append(each[0])

# # Create a trace
# trace = go.Scatter(
#     x = weather_bin.Date,
#     mode = "lines",
#     y = weather_bin.MeanTemp,
#     marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
#     name = "Mean Temperature"
# )
# trace1 = go.Scatter(
#     x = aerial_war["Mission Date"],
#     mode = "markers",
#     y = aa,
#     marker = dict(color = 'rgba(16, 0, 200, 1)'),
#     name = "Bombing temperature"
# )
# layout = dict(title = 'Mean Temperature --- Bombing Dates and Mean Temperature at this Date')
# data = [trace,trace1]
#
# fig = dict(data = data, layout = layout)
# #py.plot(fig)
# #https://plot.ly/~Anagha99/10/mean-temperature-bombing-dates-and-mean-temperature-at-this-date/#/
# * Green line is mean temperature that is measured in Bindukuri.
# * Blue markers are bombing dates and bombing date temperature.
# * As it can be seen from plot, USA bomb at high temperatures.
#     * The question is that can we predict future weather and according to this prediction can we know whether bombing will be done or not.
#     * In order to answer this question lets first start with time series prediction.
##########################################################################################
##########################################################################################
##########################################################################################
#######################################Time Series Prediction using ARIMA#################
###########################################################################################
# Notes on Time Series ARIMA:

# What is time series?
# * Time series is a collection of data points that are collected at constant time intervals.
# * It is of two types: the one which has seasonality trends & other which does not

#Stationarity of a time series:
#https://people.duke.edu/~rnau/411diff.htm
#https://stats.stackexchange.com/questions/19715/why-does-a-time-series-have-to-be-stationary

# Mean temperature of Bindikuri area
# plt.figure(figsize=(12,8))
# plt.plot(weather_bin.Date,weather_bin.MeanTemp)
# plt.title("Mean Temperature of Bindukuri Area")
# plt.xlabel("Date")
# plt.ylabel("Mean Temperature")
# plt.show()

# lets create time series from weather
timeSeries = weather_bin.loc[:, ["Date","MeanTemp"]]
#Making Date as the index & removing the Date column
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

#Now we check the stationarity of the time series.
#Two methods are used:
#Plotting Rolling Statistics: We have a window lets say window size is 6 and then we find rolling mean and variance to check stationary.

#Dickey-Fuller Test: The test results comprise of a **Test Statistic** and some **Critical Values** for difference confidence levels. If the **test statistic** is less than the **critical value**, we can say that time series is stationary.

# adfuller library
from statsmodels.tsa.stattools import adfuller


# check_adfuller
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print(result)
    print('Test statistic: ', result[0])
    print('p-value: ', result[1])
    print('Critical Values:', result[4])


# check_mean_std
def check_mean_std(ts):
    # Rolling statistics
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22, 10))
    orig = plt.plot(ts, color='red', label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label='Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    #plt.show()


# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts)
check_adfuller(ts.MeanTemp)

# * Our first criteria for stationary is constant mean. So we fail because mean is not constant as you can see from plot(black line) above . (no stationary)
# * Second one is constant variance. It looks like constant. (yes stationary)
# * Third one is that If the **test statistic** is less than the **critical value**, we can say that time series is stationary. Lets look:
#     * test statistic = -1.4 and critical values = {'1%': -3.439229783394421, '5%': -2.86545894814762, '10%': -2.5688568756191392}. Test statistic is bigger than the critical values. (no stationary)
# * As a result, we sure that our time series is not stationary.


# * Lets make time series stationary at the next part.

# ### Make a Time Series Stationary?
# * As we mentioned before, there are 2  reasons behind non-stationarity of time series
#     * Trend: varying mean over time. We need constant mean for stationary of time series.
#     * Seasonality: variations at specific time. We need constant variations for stationary of time series.


#First solving the Trend Problem:
#Solution: Moving Averages
# Moving average method
# Moving average method
window_size = 6
moving_avg = ts.rolling(window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(ts, color = "red",label = "Original")
plt.plot(moving_avg, color='black', label = "moving_avg_mean")
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
#plt.show()

ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True) # first 6 is nan value due to window size

# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_moving_avg_diff)
check_adfuller(ts_moving_avg_diff.MeanTemp)

# differencing method
ts_diff = ts - ts.shift()
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method")
plt.xlabel("Date")
plt.ylabel("Differencing Mean Temperature")
#plt.show()
ts_diff.dropna(inplace=True) # due to shifting there is nan values
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_diff)
check_adfuller(ts_diff.MeanTemp)

# * Constant mean criteria: mean looks like constant as you can see from plot(black line) above . (yes stationary)
# * Second one is constant variance. It looks like constant. (yes stationary)
# * The test statistic is smaller than the 1% critical values so we can say with 99% confidence that this is a stationary series. (yes stationary)

#Forecasting: ARIMA: auto regressive integrated moving averages
# prediction method is ARIMA that is Auto-Regressive Integrated Moving Averages.
#     * AR: Auto-Regressive (p): AR terms are just lags of dependent variable. For example lets say p is 3, we will use  x(t-1), x(t-2) and x(t-3) to predict x(t)
#     * I: Integrated (d): These are the number of nonseasonal differences. For example, in our case we take the first order difference. So we pass that variable and put d=0
#     * MA: Moving Averages (q): MA terms are lagged forecast errors in prediction equation.
# * (p,d,q) is parameters of ARIMA model.


#Choosing the p,d,q:
# ACF and PACF
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')
# ACF
plt.figure(figsize=(22,10))

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#plt.show()

# * Two dotted lines are the confidence interevals. We use these lines to determine the ‘p’ and ‘q’ values
#     * Choosing p: The lag value where the PACF chart crosses the upper confidence interval for the first time. p=1.
#     * Choosing q: The lag value where the ACF chart crosses the upper confidence interval for the first time. q=1.
# * Now lets use (1,0,1) as parameters of ARIMA models and predict
#     * ARIMA: from statsmodels libarary
#     * datetime: we will use it start and end indexes of predict method
# ARIMA LİBRARY
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime

# fit model
model = ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
print(ts.index)
model_fit = model.fit(disp=0)
forecast = model_fit.predict()
#model.predict()
# predict
#start_index = '1944-06-25'
#end_index = '1945-05-31'
#start_index = datetime(1944, 6, 25).strftime('%Y-%m-%d')
#end_index = datetime(1945, 5, 31).strftime('%Y-%m-%d')
#print(start_index)
#forecast = model.predict(model_fit,start=start_index,end = end_index)
#forecast = model_fit.predict(start = 999 ,end= 1200)
#
# # visualization
plt.figure(figsize=(12,8))
plt.plot(weather_bin.Date,weather_bin.MeanTemp,label = "original")
plt.plot(forecast,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.show()

# #Mean squared error
# # predict all path
from sklearn.metrics import mean_squared_error
# fit model
model2 = ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit2 = model2.fit(disp=0)
forecast2 = model_fit2.predict()
error = mean_squared_error(ts, forecast2)
print("error: " ,error)
# visualization
plt.figure(figsize=(12,8))
plt.plot(weather_bin.Date,weather_bin.MeanTemp,label = "original")
plt.plot(forecast2,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.savefig('graph.png')

plt.show()