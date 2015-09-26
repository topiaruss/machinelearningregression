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


df = pd.read_csv('./data/bikes_subsampled.csv')
temperature = df[['temperature']].values
bikes = df['count'].values
for degree in range(1, 11):

    polynomial_regression = PolynomialRegression(degree=degree)
    scores_cv = cross_val_score(polynomial_regression, temperature, \
                          bikes, scoring = 'mean_absolute_error', cv=5)
    score_cv_m = -np.mean(scores_cv)
    plt.plot(degree, score_cv_m, 'bo')
plt.ylabel('cross val score')
plt.xlabel('poly degree')
plt.show()
