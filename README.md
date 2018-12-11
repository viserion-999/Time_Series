# Time_Series Forecasting using Arima
# Predicting the attack during World War 2 using climate data

In this project, we are trying to predict if an attack is likely to occur & if so who is the attacker. We are predicting based on temperature of the region

After cleaning and doing the basic EDA
 we plot & visualize the attackers,target & what is the mean temperature in the region
Here are some sample plots:
https://plot.ly/~Anagha99/8/bombing-paths-from-attacker-country-to-target/#/
 https://plot.ly/~Anagha99/6/countries-take-off-bases/#/
 
 We only consider the USA-Burma war & choose Bindukuri as the weather station.
 Based on the visualization we observe that most of the attacks by US in Bindukuri area occurued during high temperatures.
 
 
We use ARIMA to predict the future temperature to decide if US will attack in the future.

Beautiful Notes on ARIMA:
#https://people.duke.edu/~rnau/411diff.htm
#https://stats.stackexchange.com/questions/19715/why-does-a-time-series-have-to-be-stationary

To test stationarity of time series we use
1)Rolling Statistics
2)Dickey Fuller's Test


Once verifying that the data is not stationary, we use moving averages to make it stationary & then we fit our model
we use ACF & PACF to decide p,d,q variables for the model.

Apologies for not making a proper notebook. _/\_
I know how big a crime it is!
Most of the code is self explanatory & has comments.

Completely referred & practised from this kaggle kernel: https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda
