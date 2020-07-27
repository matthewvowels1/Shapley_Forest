import pandas as pd
import os
import shap
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, \
    RandomizedSearchCV
import joblib


# some material taken from
# github.com/manujosephv/interpretability_blog/blob/master/census_income_interpretability.ipynb

class RFShap(object):
    def __init__(self, model_dir=None, exclude_vars=None, outcome_var=None, output_dir=None, random_seed=42,
                 class_='RF', type_='reg', balanced='balanced', trn_tst_split=0.6):

        '''
        :param model_dir: this is for pre_trained model loading
        :param exclude_vars: list the variables to exclude from the analysis (if any)
        :param outcome_var: name the outcome variable
        :param output_dir: name the directory to store results
        :param random_seed: set random seed
        :param class_: either 'RF' for random forest, or 'lin' for linear/logistic
        :param type_:  either 'reg' or 'cls' for regression or classification
        :param balanced:  either 'balanced' or 'unbalanced' for imblearn or sklearn respectively
        '''
        assert class_ in ['RF', 'lin'], 'Class not recognised - choose reg or lin.'
        assert type_ in ['reg', 'cls'], 'Type not recognised - choose reg or cls.'
        assert balanced in ['balanced', None], 'Balanced must be balanced or unbalanced'
        assert outcome_var is not None, 'Pick an outcome variable!'

        self.model_dir = model_dir
        self.exclude_vars = exclude_vars
        self.outcome_var = outcome_var
        self.output_dir = output_dir
        self.seed = random_seed
        self.class_ = class_
        self.type_ = type_
        self.balanced = balanced
        self.trn_tst_split = trn_tst_split
        self.dataset = None
        self.model = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None

    def munch(self, dataset):
        '''
        :param dataset:
        :return: self.dataset, self.X (predictors), self.y (targets), self.X_train, self.X_test, self.y_train, self.y_test
        '''

        print('Preparing dataset...')
        self.dataset = dataset.drop(columns=self.exclude_vars)
        self.y = self.dataset[[self.outcome_var]]
        self.X = self.dataset.drop(columns=self.outcome_var)

        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

        return self.dataset,  self.X,  self.y, self.X_train, self.X_test, self.y_train, self.y_test

    def make_model(self, config=None):
        '''
        :param config : model parameters
        :return: self.model
        '''

        print('Creating fresh model...')

        if self.class_ == 'RF':
            if self.type_ == 'reg':
                if self.balanced == 'balanced':
                    print('WARNING: balanced regressor not applicable')
                    self.model = RandomForestRegressor(**config) if config != None else RandomForestRegressor()
                elif self.balanced == None:
                    self.model = RandomForestRegressor(**config) if config != None else RandomForestRegressor()
            elif self.type_ == 'cls':
                if self.balanced == 'balanced':
                    self.model = BalancedRandomForestClassifier(**config) if config != None else BalancedRandomForestClassifier()
                elif self.balanced == None:
                    self.model = RandomForestClassifier(**config) if config != None else RandomForestClassifier()
        elif self.class_ == 'lin':
            if self.type_ == 'reg':
                if self.balanced == 'balanced':
                    print('WARNING: balanced regressor not applicable')
                    self.model = LinearRegression(**config) if config != None else LinearRegression()
                elif self.balanced == None:
                    self.model = LinearRegression(**config) if config != None else LinearRegression()
            elif self.type_ == 'cls':
                if self.balanced == 'balanced':
                    self.model = LogisticRegression(**config) if config != None else LogisticRegression()
                    self.model.class_weight = self.balanced
                elif self.balanced == None:
                    self.model = LogisticRegression(**config) if config != None else LogisticRegression()
                    self.model.class_weight = None
        print('Created: ', self.model)
        return self.model


    def _split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=self.trn_tst_split, random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def train_test(self):

        '''
        :return: trained model and performance report
        '''

        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'

        self.model.fit(self.X_train, self.y_train)
        report = self._eval()
        return self.model, report

    def _eval(self):

        '''
        :return: performance report
        '''
        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'

        preds = self.model.predict(self.X_test)
        df_report = None

        if self.type_ == 'cls':
            print("Accuracy: %s%%" % (100 * metrics.accuracy_score(self.y_test, preds)))
            conf_mat = np.asarray(metrics.confusion_matrix(self.y_test, preds))
            np.savetxt(os.path.join(self.output_dir, 'simple_train_test_conf_mat.txt'), conf_mat)
            report = metrics.classification_report(self.y_test, preds, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(os.path.join(self.output_dir,  str(self.type_) + '_' + str(self.class_) +
                                          '_simple_train_test_classification_report.csv'))

        elif self.type_ == 'reg':
            expl_var = metrics.explained_variance_score(self.y_test, preds)
            mae = metrics.mean_absolute_error(self.y_test, preds)
            mse = metrics.mean_squared_error(self.y_test, preds)
            msle = metrics.mean_squared_log_error(self.y_test, preds)
            med_ae = metrics.median_absolute_error(self.y_test, preds)
            r2 = metrics.r2_score(self.y_test, preds)
            cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
            df_report = pd.DataFrame([expl_var, mae, mse, msle, med_ae, r2]).T
            df_report.columns = cols
            df_report.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                          '_simple_train_test_regression_report.csv'))

        return df_report

    def tune_model(self, tunable_params=None, folds=3, n_iter=100):
        '''
        :param tunable_params: list of desired parameters to tune over
        :param folds: number of folds for CV
        :param n_iter: number of iters for tuning
        :return: model with optimal hyperparameters
        '''
        print('Tuning the following parameters: ', tunable_params)

        if self.class_ == 'RF':
            possible_params = ['n_estimators', 'criterion', 'max_features', 'max_leaf_nodes', 'max_depth',
                               'min_samples_leaf', 'min_samples_split', 'bootstrap']

            for param in tunable_params:
                if param not in possible_params:
                    print('WARNING: ', param, ' not a tunable parameter for this model.')

            n_estimators = max_features = max_depth = criterion = min_samples_split = bootstrap = min_samples_leaf = max_leaf_nodes = None

            if 'n_estimators' in tunable_params:
                n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
            if 'criterion' in tunable_params:
                criterion = ['mse', 'mae']
            if 'max_features' in tunable_params:
                max_features = ['auto', 'sqrt']
            if 'max_leaf_nodes' in tunable_params:
                max_leaf_nodes = [0, 1, 2, 3, 4, 5]
            if 'max_depth' in tunable_params:
                max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                max_depth.append(None)
            if 'min_samples_split' in tunable_params:
                min_samples_split = [2, 5, 10]
            if 'min_samples_leaf' in tunable_params:
                min_samples_leaf = [1, 2, 4]
            if 'bootstrap' in tunable_params:
                bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'criterion': criterion,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'max_leaf_nodes': max_leaf_nodes,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}

        if self.class_ == 'lin':

            if self.type_ == 'reg':
                possible_params = ['normalize', 'fit_intercept']

                for param in tunable_params:
                    if param not in possible_params:
                        print('WARNING: ', param, ' not a tunable parameter for this model.')

                normalize = fit_intercept = None

                if 'normalize' in tunable_params:
                    normalize = [True, False]
                if 'fit_intercept' in tunable_params:
                    fit_intercept = [True, False]

                # Create the random grid
                random_grid = {'normalize': normalize,
                               'fit_intercept': fit_intercept}

            elif self.type_ == 'cls':
                possible_params = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'solver', 'max_iter']

                for param in tunable_params:
                    if param not in possible_params:
                        print('WARNING: ', param, ' not a tunable parameter for this model.')

                penalty = dual = tol = C = fit_intercept = solver = max_iter = None

                if 'penalty' in tunable_params:
                    penalty = ['l1', 'l2', 'elasticnet', None]
                if 'dual' in tunable_params:
                    dual = [True, False]
                if 'tol' in tunable_params:
                    tol = [1e-2, 1e-3, 1e-4, 1e-5]
                if 'C' in tunable_params:
                    C = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                if 'fit_intercept' in tunable_params:
                    fit_intercept = [True, False]
                if 'solver' in tunable_params:
                    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                if 'max_iter' in tunable_params:
                    max_iter = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

                # Create the random grid
                random_grid = {'penalty': penalty,
                               'fit_intercept': fit_intercept,
                               'dual': dual,
                               'tol': tol,
                               'C': C,
                               'solver': solver,
                               'max_iter': max_iter,
                               }
        print(random_grid)
        filtered = {k: v for k, v in random_grid.items() if v is not None}
        random_grid.clear()
        random_grid.update(filtered)
        print(random_grid)
        rf = self.make_model(config=None)
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=folds, verbose=2,
                                       random_state=self.seed, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)
        rf_params = rf_random.best_params_
        print('best params are: ', rf_params)
        self.model = self.make_model(config=rf_params)

        return self.model


    def run_shap_explainer(self, modell):

        '''
        :param modell: give it a trained model
        :return:
        '''
        assert modell is not None, 'Feed me a model : ]'

        print('Running Shap Explainer using the same train/test split you used to train the model.')
        explainer = shap_vals = None
        if self.class_ == 'RF':
            explainer = shap.TreeExplainer(model=modell, model_output='margin')
            shap_vals = explainer.shap_values(self.X_test)

            joblib.dump(explainer, os.path.join(self.output_dir, self.outcome_var + "_tree_explainer.sav"))
            joblib.dump(shap_vals, os.path.join(self.output_dir, self.outcome_var + "_tree_explainer_shap_values.sav"))
        else:
            explainer = shap.LinearExplainer(modell, self.X_test)
            shap_vals = explainer.shap_values(self.X_test)
            joblib.dump(explainer, os.path.join(self.output_dir, self.outcome_var + "_linear_explainer.sav"))
            joblib.dump(shap_vals, os.path.join(self.output_dir, self.outcome_var + "_linear_explainer_shap_values.sav"))

        return explainer, shap_vals


    def shap_plot(self, shap_vals=None, specific_var=None, interaction_var=None, classwise=True, class_ind=1):

        '''
        :param explainer: explainer
        :param shap_vals: vals derived from running the explainer
        :param specific_var: if desired, run the individual feature plots
        :param interaction_var: which desired var to plot as interacting with 'specific var'
        :param class ind: when plotting classifier results, pick class index to plot with
        '''

        if self.type_ == 'cls':

            if classwise or (self.class_ == 'lin'):
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='bar', show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_val_summary.png'))
                plt.show()
                plt.close()
            else:
                shap.summary_plot(shap_values=shap_vals[class_ind], features=self.X_test, max_display=20, plot_type='bar', show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_val_summary.png'))
                plt.show()
                plt.close()

            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals[class_ind], features=self.X_test, max_display=20, plot_type='dot',
                              show=False)
            else:
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='dot',
                              show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                     shap_values=shap_vals[class_ind], features=self.X_test)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                             shap_values=shap_vals, features=self.X_test)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_effects_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()

        elif self.type_ == 'reg':

            shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='bar',
                              show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_val_summary.png'))
            plt.show()
            plt.close()

            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='dot',
                                  show=False)
            else:
                shap.summary_plot(shap_values=shap_vals[0], features=self.X_test, max_display=20, plot_type='dot',
                          show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals, features=self.X_test)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals[0], features=self.X_test)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) + 'shap_effects_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()
















