import tensorflow as tf
import numpy as np
import pandas as pd

df_train = pd.read_csv('./data/train.csv')
# drop useless data
df_train = df_train.drop(['Street', 'LotFrontage', 'LandSlope', 'YearBuilt', 'YearRemodAdd',
                          'MasVnrArea', 'Foundation', 'GarageYrBlt', 'MoSold', 'YrSold'], 1)