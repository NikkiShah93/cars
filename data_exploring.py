#In this part we will explore our data
#We are trying to find the best feature to use in our model
#There are different things that could be useful when you are exploring the data
#One of them is correlation between the feature and our target
#Other things like ANOVA could be considered for non numeric variables

#Most of the codes here are commented out but you could easily take a look at graphs and tables as you go through

#We use Numpy, Pandas, Stats from Scipy, Seaborn and Matplotlib

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

#First we will import the file that we created in the data cleaning step
cars_data = pd.read_csv('cars_dataset.csv')

#We could take a quick look at the correlation between all the numerical variables using Pandas data frame corr function
#It provides a table that the values in its diagonal is 1, which shows th correlation between a variable with itself

#cars_data.corr()

#After checking the correlation table we could say that:
#Curb weight (0.83), Engine size (0.87), Highway-mpg (-0.70), Horse power (0.80), Width (0.75) and Lenght (0.69)
#have strong corraletion with price

#Other ways to check for correlation is by visualization
#Seaborn regplot could be very useful for that for numerical variables

#sns.regplot(x = 'Engine-size', y = 'Price', data = cars_data)

#This plot shows the strong positive correlation between Engine size and Price

#sns.regplot(x = 'Curb-weight', y = 'Price', data = cars_data)

#This plot shows the strong positive correlation between Curb weight and Price

#For categorical data we could use boxplot to see the correlation

#sns.boxplot(x = 'Number-of-doors', y = 'Price', data = cars_data)

#The distribution of two types of number of doors are very similar so it could not be a good predictor

#sns.boxplot(x = 'Fuel-type', y = 'Price', data = cars_data)

#The distribution of two fuel types are very similar so it could not be a good predictor

#sns.boxplot(x = 'Drive-wheel', y = 'Price', data = cars_data)

#The distribution of different drive wheels are not that similar so it could be a good predictor

#The other way to test for correlation is by using a piviot table with the variables that we want to take a look at

grouped_data = cars_data[["Drive-wheel", "body-style", "Price"]].groupby(["Drive-wheel", "body-style"], as_index = False).mean()
piviot_table = grouped_data.pivot(index = "body-style", columns="Drive-wheel")

#Loking at this piviot table shows that rwd cars are more expensive than the other two types in every body style
#We could use a heat map to visualize it better, but there are nan values in our pivit that has to be filled

piviot_table = piviot_table.fillna(0)

#We use seaborn.heatmap for generating the heatmap

sns.heatmap(piviot_table)
plt.show()