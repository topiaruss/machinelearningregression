from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


bikes_df = pd.read_csv('./data/bikes_subsampled.csv')

# We select the variables temperature and bikes_count
a = 28
temperature = bikes_df['temperature'].values
bikes_count = bikes_df['count'].values
temperature_predict = np.expand_dims(a=np.linspace(-5, 40, 100), axis=1)
bikes_count_predict = a*temperature_predict
plt.scatter(temperature, bikes_count, color='k')
plt.plot(temperature_predict, bikes_count_predict, linewidth=2)
plt.xlabel('temperature')
plt.ylabel('No. HIred')
plt.xlim(-5,40)
plt.tight_layout()
plt.show()

