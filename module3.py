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

temperature = bikes_df['temperature'].values
bikes_count = bikes_df['count'].values

plt.scatter(temperature, bikes_count, color='k')
plt.xlabel('temperature')
plt.ylabel('hired')
plt.xlim(-5,40)
plt.tight_layout()
plt.show()
