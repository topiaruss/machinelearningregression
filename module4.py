from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

################################################################################
################################### MODULE 4 ###################################
##################### Multiple variables linear regression #####################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

from functions import PolynomialRegression, model_plot_3d


bikes_df = pd.read_csv('./data/bikes_subsampled.csv')
bikes_df = pd.read_csv('./data/bikes.csv')

# Learning activity 1: Fit a model of 2 variables and plot the model

features = ['temperature','humidity']
X = bikes_df[features].values
y = bikes_df['count'].values


print 'Correlation coefficient for temp:', np.corrcoef(bikes_df['temperature'],
                                                       bikes_df['count'])[0, 1]
print 'Correlation coefficient for humidity:', \
    np.corrcoef(bikes_df['humidity'],
    bikes_df['count'])[0, 1]

linear_regression = LinearRegression()
linear_regression.fit(X, y)

print 'Bikes hired at 20C, and 60%: ',\
    linear_regression.predict([20, 60])

print 'Bikes hired at 5, and 90%: ',\
    linear_regression.predict([5, 90])

bikes_count = bikes_df['count'].values
temperature = bikes_df[['temperature']].values
polynomial_regression = PolynomialRegression(degree=2)
polynomial_regression.fit(temperature, bikes_count)
temperature_predict = np.expand_dims(np.linspace(-5, 40, 100), 1)
bikes_count_predict = polynomial_regression.predict(temperature_predict)

plt.scatter(temperature, bikes_count, color='k')
plt.plot(temperature_predict, bikes_count_predict, linewidth=2)
plt.legend(['Data', 'Degree 2'], loc=0)
plt.show()
