# plot feature importance manually
from numpy import loadtxt
# from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_xgboost(train, test):

    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]

    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]

    data_dmatrix = xgb.DMatrix(data=train_data, label=train_target)

    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)

    xg_reg.fit(train_data, train_target)

    preds = xg_reg.predict(test_data)

    rmse = np.sqrt(mean_squared_error(test_target, preds))
    print("RMSE: %f" % (rmse))

    params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)


    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

    # fit model no training data
    #model = XGBClassifier()
   # model.fit(train[['wtd_entropy_atomic_mass', 'critical_temp']], train['critical_temp'])
    # feature importance
    #print(model.feature_importances_)
    # plot
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
   # pyplot.show()
