from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold


class GridSearchs:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.model = model
        self.grid_param ={'n_estimators':[1000,2000],'max_depth':[4,8,16],'num_leaves':[31,15,7,3],
             'learning_rate':[0.1,0.05,0.01]}
        self.fit_params={'early_stopping_rounds':10, 
            'eval_metric' : 'rmse', 
            'eval_set' : [(self.X_train, self.y_train)]
           }

        self.bst_gs_cv = GridSearchCV(
                    self.model, # model for gridsearch
                    self.grid_param, # set best hyper param
                    cv = KFold(n_splits=3, shuffle=True), # num of validation
                    scoring = 'neg_mean_squared_error',
                    verbose = 0
                    )
    def train(self):
        self.bst_gs_cv.fit(self.X_train, self.y_train,
                    **self.fit_params, verbose = 0)
        return self.bst_gs_cv
    
    def main(self):
        bst_gs_cv = self.train()
        best_param = bst_gs_cv.best_params_
        print('Best parameter: {}'.format(best_param))

        pred = bst_gs_cv.predict(self.X_test)
        RMSE = np.sqrt(mean_squared_error(pred, self.y_test))
        print('GridSearchCV RMSE:{}'.format(RMSE))
