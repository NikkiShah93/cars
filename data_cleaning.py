
# coding: utf-8

# In[1]:

#We need to import the dataset first
#It's an online dataset from 1980s storing features of cars and their prices
#Dataset is old and messed up so we have to correct the problems then use it
#The main goal is to find a regression model that could predict the price of cars having some features

#The libraries that we use here are, Numpy, Scipy, Scikit-learn, Pandas, Seaborn and Matplotlib
#For this file only Pandas and Numpy are useful

import numpy as np
import pandas as pd


#Importing the dataset

#The url of the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

#This dataset does not have any headers so we put header = None
cars_dataset = pd.read_csv(url, header = None)

#For uderstanding the data better we need to have headers for our data frame
headers = ["Symboling", "Normalized-losses", "Make", "Fuel-type", "Asprition", "Number-of-doors", "body-style", "Drive-wheel", "Engine-location", "Wheel-base", "length", "Width", "Height", "Curb-weight", "Engine-type", "Number-of-cylenders",
          "Engine-size", "Fuel-system", "Bore", "Stroke", "Compression-ratio", "Horse-power", "Peak-rpm", "City-mpg", "Highway-mpg","Price"]

#Using the header list for name of the columns
cars_dataset.columns = headers

#After taking a look at the data frame, we notice that there are '?' values in some of our cells
#For correcting them easier, we replace them with Numpy NaN value
#The data types are messed up too, some of the columns have to have numeric types instead of object type

cars_dataset.replace('?', np.nan, inplace = True)

#Now we have to do something about NaN values, we treat each columns differently 
#because they have different type and influence on our model

#For Normalized-losses, Bore, Stroke, Horse-power and Peak-rpm columns, we replace NaN with their mean
#For Number-of-doors, we replace NaN with the most frequent value
#For a column like Price that is our target we have to drop NaNs because the observations without price are useless

#Replacing with their mean values
cars_dataset["Normalized-losses"].replace(np.nan, cars_dataset["Normalized-losses"].astype('float').mean(), inplace = True)
cars_dataset["Bore"].replace(np.nan, cars_dataset["Bore"].astype("float").mean(), inplace = True)
cars_dataset["Stroke"].replace(np.nan, cars_dataset["Stroke"].astype("float").mean(), inplace = True)
cars_dataset["Horse-power"].replace(np.nan, cars_dataset["Horse-power"].astype("float").mean(), inplace = True)
cars_dataset["Peak-rpm"].replace(np.nan, cars_dataset["Peak-rpm"].astype("float").mean(), inplace = True)

#Replacing with the most frequent value
cars_dataset["Number-of-doors"].replace(np.nan, cars_dataset["Number-of-doors"].value_counts().idxmax(), inplace = True)

#Droping the observations with NaN values for Price column
cars_dataset.dropna(subset = ['Price'],inplace = True, axis = 0)

#After droping the rows with NaN we have to reset the indexes
cars_dataset.reset_index(drop = True, inplace=True)

#Now we have to correct the data types for each column
cars_dataset[["Price", "Bore", "Stroke", "Peak-rpm"]] = cars_dataset[["Price", "Bore", "Stroke", "Peak-rpm"]].astype('float')
cars_dataset[["Normalized-losses"]] = cars_dataset[["Normalized-losses"]].astype('int')

#After cleaning the data we could save it in a csv file in the same directory
#so we don't have to deal with cleaning each time we want to work with it
cars_dataset.to_csv('cars_dataset.csv')
