import pandas as pd
import numpy as np
import os
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import KFold

problem_title = 'Prediction of autoindustry stock fluctuation from product recalls'
_target_column_name = 'Valeur action'
Predictions = rw.prediction_types.make_regression()


class ASP(FeatureExtractorRegressor):

    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'data_co2.csv', 'voiture_groupe_marque.xlsx']):
        super(ASP, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names


workflow = ASP()

class ASP_loss(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='asp loss', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        exp_loss = np.mean(np.exp((y_true - y_pred) ** 2))
        return exp_loss

score_types = [
    ASP_loss(name='asp loss', precision=2),
]

def get_cv(X, y):
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    return cv.split(X,y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False, encoding = "ISO-8859-1")
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'recall_stock_train.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'recall_stock_test.csv'
    return _read_data(path, f_name)
