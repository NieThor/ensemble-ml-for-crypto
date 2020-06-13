import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def fit_models(df: pd.DataFrame):
    trained_models = []
    y = df['target']
    df.drop(columns=['target'], inplace=True)
    estimators = {
        'Logistic_Regression': (LogisticRegression(penalty='l2', solver='sag', max_iter=1000), {
            'C': [0.1, 0.01, 0.001, 0.0001],
        }),
        'Random_Forest': (RandomForestClassifier(n_estimators=400, min_samples_leaf=9,
                                                 min_samples_split=9, criterion='gini',
                                                 min_impurity_decrease=0.0000001, max_features='auto'), {}),
        'SVC': (SVC(C=1, kernel='linear', probability=True), {}),
        'xgb': (XGBClassifier(n_estimators=100, max_depth=3, gamma=1, learning_rate=0.01), {
            'min_child_weight': [1, 2],
            'lambda': [0, 1]
        })
    }
    for estimator, parameter in estimators.values():
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(estimator=estimator, param_grid=parameter, cv=tscv, n_jobs=-1,
                                   scoring='neg_log_loss')
        trained_models.append(grid_search.fit(df, y).best_estimator_)

    return trained_models
