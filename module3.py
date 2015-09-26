from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


bikes_df = pd.read_csv('./data/bikes.csv')

# We select the variables temperature and bikes_count


temperature = bikes_df['temperature'].values
bikes_count = bikes_df['count'].values
temperature_predict = np.expand_dims(a=np.linspace(-5, 40, 100), axis=1)

linear_regression = LinearRegression()
temperature_ = np.expand_dims(temperature, 1)
linear_regression.fit(temperature_, bikes_count)

print 'optimal slope:', linear_regression.coef_[0]
print 'optimal intercept:', linear_regression.intercept_

bikes_count_predict = linear_regression.predict(temperature_predict)
#print 'MAE:', metrics.mean_absolute_error(bikes_count, bikes_count_predict)

plt.scatter(temperature, bikes_count, color='k')
plt.plot(temperature_predict, bikes_count_predict, linewidth=2)
plt.xlabel('temperature')
plt.ylabel('No. HIred')
plt.xlim(-5,40)
plt.tight_layout()
plt.show()

