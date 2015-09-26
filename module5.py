from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

################################################################################
################################### MODULE 5 ###################################
############################# Model evaluation #################################
################################################################################
# To be separated in a unique file #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

bikes_df = pd.read_csv('./data/bikes_subsampled.csv')
#bikes_df = pd.read_csv('./data/bikes.csv')
temperature = bikes_df[['temperature']].values
bikes_count = bikes_df['count'].values


temp_train, temp_test, bikes_train, bikes_test = \
            train_test_split(temperature, bikes_count, test_size=0.5)

#plt.scatter(temp_train, bikes_train, color = 'k')
#plt.scatter(temp_test, bikes_test, color = 'r')
#plt.show()

#polynomial_regression = PolynomialRegression(degree=4)
#polynomial_regression.fit(temp_train, bikes_train)
#temperature_predict = np.expand_dims(np.linspace(-5,40,100),1)
#bikes_predict = polynomial_regression.predict(temperature_predict)
#plt.plot(temperature_predict, bikes_predict, linewidth=2)
#plt.scatter(temp_train, bikes_train, color='k')
#plt.scatter(temp_test, bikes_test, color='r')
#plt.ylim(0, 1400)
#plt.show()

df = pd.read_csv('./data/bikes.csv')
temperature = df[['temperature']].values
bikes = df['count'].values
polynomial_regression = PolynomialRegression(degree=2)
scores = -cross_val_score(polynomial_regression, temperature, \
                          bikes, scoring = 'mean_absolute_error', cv=5)

#plt.scatter(temperature, bikes, color='k')
#plt.show()
print scores
print np.mean(scores)