from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0


################################################################################
################################### MODULE 6 ###################################
############################# Regularisation ###################################
################################################################################


# Learning activity 2: Ridge and Lasso regularisations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.grid_search import GridSearchCV

from functions import PolynomialRidge, PolynomialLasso, PolynomialRegression

bikes_df = pd.read_csv('./data/bikes.csv')
features = ['temperature', 'humidity', 'windspeed']
X = bikes_df[features].values
y = bikes_df['count'].values

polynomial_ridge = PolynomialRidge(degree=2, alpha=0)
polynomial_ridge.fit(X, y)
coefs_0 = polynomial_ridge.steps[2][1].coef_
print coefs_0

polynomial_ridge = PolynomialRidge(degree=2, alpha=1)
polynomial_ridge.fit(X, y)
coefs_1 = polynomial_ridge.steps[2][1].coef_
print coefs_1
