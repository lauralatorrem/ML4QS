from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

import sys
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set up file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter2_result.csv'
RESULT_FNAME =  'chapter3_result_outliers.csv'

# Next, import the data from the specified location and parse the date index.
try:
    dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset(__file__)

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter5_result.csv'

DataViz = VisualizeDataset(__file__)

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

# And now some example code for using the dynamical systems model with parameter tuning (note: focus on predicting accelerometer data):

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate', '2016-02-08 18:29:56',
#                                                                                   '2016-02-08 18:29:58','2016-02-08 18:29:59')
                                                                                   '2016-02-08 19:34:07', '2016-02-08 20:07:50')

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Now let us focus on the learning part.

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

dataset['activityLevel'] = 0
dataset.loc[dataset['labelOnTable'] == 1 or dataset['labelSitting'] == 1 or dataset['labelWashingHands'] == 1 or dataset['labelStanding'] == 1, 'activityLevel'] = 'Low'
dataset.loc[dataset['labelWalking'] == 1 or dataset['labelDriving'] == 1 or dataset['labelEating'] == 1, 'activityLevel'] = 'Medium'
dataset.loc[dataset['labelRunning'] == 1, 'activityLevel'] = 'High'

selected_features = ['self.activityLevel', 'self.hr_watch_rate', 'self.acc_phone_y_temp_mean_ws_120']

output_sets = learner.dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y, selected_features,
                                                     ['self.a * self.activityLevel + self.b * self.hr_watch_rate',
                                                      'self.c * self.hr_watch_rate + self.d * self.acc_phone_y_temp_mean_ws_120',
                                                      'self.e * self.activityLevel + self.f * self.acc_phone_y_temp_mean_ws_120'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=10, max_generations=10, per_time_step=True)
DataViz.plot_pareto_front(output_sets)

DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y, output_sets, 0, 'hr_watch_rate')

regr_train_y, regr_test_y = learner.dynamical_systems_model_ga(train_X, train_y, test_X, test_y, selected_features,
                                                     ['self.a * self.activityLevel + self.b * self.hr_watch_rate',
                                                      'self.c * self.hr_watch_rate + self.d * self.acc_phone_y_temp_mean_ws_120',
                                                      'self.e * self.activityLevel + self.f * self.acc_phone_y_temp_mean_ws_120'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=5, max_generations=10, per_time_step=True)

DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['hr_watch_rate'], regr_train_y['hr_watch_rate'], test_X.index, test_y['hr_watch_rate'], regr_test_y['hr_watch_rate'], 'hr_watch_rate')

regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X, train_y, test_X, test_y, selected_features,
                                                     ['self.a * self.activityLevel + self.b * self.hr_watch_rate',
                                                      'self.c * self.hr_watch_rate + self.d * self.acc_phone_y_temp_mean_ws_120',
                                                      'self.e * self.activityLevel + self.f * self.acc_phone_y_temp_mean_ws_120'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     max_generations=10, per_time_step=True)

DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['hr_watch_rate'], regr_train_y['hr_watch_rate'], test_X.index, test_y['hr_watch_rate'], regr_test_y['hr_watch_rate'], 'hr_watch_rate')
