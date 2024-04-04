from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna.integration.lightgbm as optuna_lgb


class optuna_lightBGMs:
    def __init__(self, params, X_train, X_test, y_train, y_test):
        self.params = params
        self.lgb_train = optuna_lgb.Dataset(X_train, y_train)
        self.lgb_eval = optuna_lgb.Dataset(X_test, y_test, reference=self.lgb_train)   
        # save training history
        self.best_params, self.evaluation_results = {}, {}

        
    def train(self):
        gbm = optuna_lgb.train(self.params,
                        self.lgb_train,
                        num_boost_round=1000,
                        valid_names=['train', 'valid'],
                        valid_sets=[self.lgb_train, self.lgb_eval],
                        evals_result=self.evaluation_results, 
                        early_stopping_rounds=50
                       )

        best_params = gbm.params
        print(best_params)
        self.history_plot(save_name="lbgm_optuna_history.jpg")
        return gbm, best_params
    
    def history_plot(self, save_name="lbgm_optuna_history.jpg"):
        plt.plot(self.evaluation_results['train']['rmse'], label='train')
        plt.plot(self.evaluation_results['valid']['rmse'], label='valid')
        plt.ylabel('Log loss')
        plt.xlabel('Boosting round')
        plt.title('Training performance')
        plt.legend()
        plt.savefig(save_name)
        plt.show()
        

class lightGBMs:
    def __init__(self, bgm_model, X_train, X_test, y_train, y_test):
        # lightGBM model
        self.bst = bgm_model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        
    def train(self):
        self.bst.fit(self.X_train, self.y_train,
                  eval_names=['train'],
                  eval_set=[(self.X_test.values, self.y_test.values)],
                  early_stopping_rounds=50,  
                  eval_metric='rmse',
                  verbose = 0)
        
        pred = self.bst.predict(self.X_test)
        RMSE = np.sqrt(mean_squared_error(pred, self.y_test))
        print('default RMSE:{}'.format(RMSE))
        return self.bst
    
    def plotImp(self, model, columns_list, fig_size = (40, 20), save_name='lgbm_feature_imp.png', 
              on_display=False):
        num = len(columns_list)
        if on_display:
            importance = pd.DataFrame(model.feature_importances_, index=columns_list, columns=['importance'])
            importance = importance.sort_values('importance', ascending=False)
            display(importance)
        feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':columns_list})
        plt.figure(figsize=fig_size)
        sns.set(font_scale = 5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                            ascending=False)[0:num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(save_name)
        plt.show()


def calculate_scores(true, pred):
    scores = {}
    scores = pd.DataFrame({'R2': r2_score(true, pred),
                          'MAE': mean_absolute_error(true, pred),
                          'MSE': mean_squared_error(true, pred),
                          'RMSE': np.sqrt(mean_squared_error(true, pred))},
                           index = ['scores'])
    return scores

def plotImp(model, columns_list, fig_size = (40, 20), save_name='lgbm_feature_imp.png', on_display=False):
    num = len(columns_list)
    if on_display:
        importance = pd.DataFrame(model.feature_importance(), index=columns_list, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        display(importance)
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':columns_list})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()