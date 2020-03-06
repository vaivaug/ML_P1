# plot feature importance manually
from numpy import loadtxt
# from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import math


def run_xgboost(train, test):

    '''
    train_target = train[['critical_temp']]
    train_data = train[correlated_features]

    test_target = test[['critical_temp']]
    test_data = test[correlated_features]
    '''

    i=1
    mses = []

    for i in range(1, 26):
        i += 1
        train = train.reset_index(drop=True)
        testing = train.sample(frac=0.3)
        train = train.drop(testing.index)

        train_target = train.iloc[:, -1]
        train = train.iloc[:, :-1]

        test_target = test.iloc[:, -1]
        test = test.iloc[:, :-1]

        model = xgb.XGBRegressor()
        model.fit(train, train_target)

        preds = model.predict(test)

        mses.append(mean_squared_error(test_target, preds))


    print(len(mses))
    average_mses = sum(mses)/len(mses)

    print('square root of this average: ', math.sqrt(average_mses))



    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]

    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]


    #xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
     #                         max_depth=5, alpha=10, n_estimators=10)

    D_train = xgb.DMatrix(data=train_data, label=train_target)
    D_test = xgb.DMatrix(test_data, label=test_target)

    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': 3}

    steps = 25

  #  scores = cross_val_score(model, train_data, train_target, cv=5)
   # print("Mean cross-validation score: %.2f" % scores.mean())

    '''
    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(model, train_data, train_target, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    preds = model.predict(test_data)
    mse = mean_squared_error(test_target, preds)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % np.sqrt(mse))

    x_ax = range(len(test_target))
    plt.scatter(x_ax, test_target, s=5, color="blue", label="original")
    plt.plot(x_ax, preds, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
    '''

    # print(preds)
    # print(test_target)
    '''
    
    data_dmatrix = xgb.DMatrix(data=train_data, label=train_target)
        
    rmse = np.sqrt(mean_squared_error(test_target, preds))
    print("RMSE: %f" % (rmse))

    params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    '''
    #xgb.plot_importance(xg_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    #plt.show()

    # fit model no training data
    #model = XGBClassifier()
   # model.fit(train[['wtd_entropy_atomic_mass', 'critical_temp']], train['critical_temp'])
    # feature importance
    #print(model.feature_importances_)
    # plot
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
   # pyplot.show()
