import pandas as pd
import os
import shap
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import sem
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, LeaveOneOut
import joblib

EPS = 1e-3
# some material taken from
# github.com/manujosephv/interpretability_blog/blob/master/census_income_interpretability.ipynb

class RFShap(object):
    def __init__(self, model_dir=None, exclude_vars=None, outcome_var=None, output_dir=None, random_seed=42,
                 class_='RF', type_='reg', balanced='balanced', trn_tst_split=0.6, k_cv='loo_cv', k=5):

        '''
        :param model_dir: this is for pre_trained model loading
        :param exclude_vars: list the variables to exclude from the analysis (if any)
        :param outcome_var: name the outcome variable
        :param output_dir: name the directory to store results
        :param random_seed: set random seed
        :param class_: either 'RF' for random forest, or 'lin' for linear/logistic
        :param type_:  either 'reg' or 'cls' for regression or classification
        :param balanced:  either 'balanced' or 'unbalanced' for imblearn or sklearn respectively
        :param k_cv: uses k_fold training across entire dataset (but does not allow hyperparam tuning).
        :param k: if k_cv='loo_cv', 'split', 'k_fold'
        '''
        assert class_ in ['RF', 'lin'], 'Class not recognised - choose reg or lin.'
        assert type_ in ['reg', 'cls'], 'Type not recognised - choose reg or cls.'
        assert balanced in ['balanced', None], 'Balanced must be balanced or unbalanced'
        assert outcome_var is not None, 'Pick an outcome variable!'
        assert k_cv in ['loo_cv', 'k_fold', 'split'], "For k_cv Choose from 'loo_cv', 'k_fold', 'split'"

        self.model_dir = model_dir
        self.exclude_vars = exclude_vars
        self.outcome_var = outcome_var
        self.output_dir = output_dir
        self.seed = random_seed
        self.class_ = class_
        self.type_ = type_
        self.balanced = balanced
        self.trn_tst_split = trn_tst_split
        self.k_cv = k_cv
        self.k = k
        self.dataset = None
        self.model = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.config = None
        self.n_categories = None

        if self.model_dir is not None:
            self.model = self.load_model(model_dir_=self.model_dir)

    def munch(self, dataset):
        '''
        :param dataset:
        :return: self.dataset, self.X (predictors), self.y (targets), self.X_train, self.X_test, self.y_train, self.y_test
        '''

        print('Preparing dataset...')
        self.dataset = dataset
        if self.exclude_vars is not None:
            self.dataset = dataset.drop(columns=self.exclude_vars)
        self.y = self.dataset[[self.outcome_var]]
        self.n_categories = len(np.unique(self.y))
        self.X = self.dataset.drop(columns=self.outcome_var)

        if self.class_ == 'lin':
            self.X = (self.X-self.X.mean())/(self.X.std() + EPS)

        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

        return self.dataset,  self.X,  self.y, self.X_train, self.X_test, self.y_train, self.y_test

    def load_model(self, model_dir_=None):
        model_dir = model_dir_

        if model_dir is None:
            model_dir = self.model_dir
        else:
            self.model_dir = model_dir_

        try:
            self.model = joblib.load(model_dir)
        except:
            print('Either no model dir specified, or model dir is incorrect (i.e. no /.../model.sav found).')

        return self.model

    def save_model(self, model_dir_=None):

        model_dir = model_dir_
        if model_dir is None:
            model_dir = self.model_dir
        else:
            self.model_dir = model_dir_
        try:
            joblib.dump(self.model, model_dir)
        except:
            print('Check model dir... had problems saving /.../model.sav.')


    def make_model(self, config=None):
        '''
        :param config : model parameters
        :return: self.model
        '''

        if config != None:
            self.config = config

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
        if self.trn_tst_split < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                test_size=self.trn_tst_split, random_state=self.seed)
        elif self.trn_tst_split == 1.0:
            X_train = self.X
            X_test = self.X
            y_train = self.y
            y_test = self.y
        return X_train, X_test, y_train, y_test

    def train_test(self, plot=False):

        '''
        :param plot: whether to plot the auroc with k-fold training (default is false)
        :return: trained model and performance report
        '''

        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'
        report = None
        if self.k_cv == 'k_fold' or self.k_cv == 'loo_cv':
            report = self._train_eval_kloo(plot=plot)

        elif self.k_cv == 'split':
            self.model.fit(self.X_train, self.y_train.values.ravel())
            report = self._eval_split()
        return self.model, report

    def _eval_split(self):

        '''
        :return: performance report
        '''
        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'

        preds = self.model.predict(self.X_test)
        df_report = None

        if self.type_ == 'cls':
            print("Accuracy: %s%%" % (100 * metrics.accuracy_score(self.y_test, preds)))
            conf_mat = np.asarray(metrics.confusion_matrix(self.y_test.values.ravel(), preds))
            np.savetxt(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' + str(self.class_) +
                                    '_simple_test_conf_mat.txt'), conf_mat)
            report = metrics.classification_report(self.y_test, preds, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(os.path.join(self.output_dir,  str(self.type_) + '_' + str(self.class_) +
                                          '_simple_test_classification_report.csv'))

        elif self.type_ == 'reg':
            expl_var = metrics.explained_variance_score(self.y_test, preds)
            mae = metrics.mean_absolute_error(self.y_test, preds)
            mse = metrics.mean_squared_error(self.y_test, preds)
            msle = 0
            try:
                msle = metrics.mean_squared_log_error(self.y_test, preds)
            except:
                pass
            med_ae = metrics.median_absolute_error(self.y_test, preds)
            r2 = metrics.r2_score(self.y_test, preds)
            cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
            df_report = pd.DataFrame([expl_var, mae, mse, msle, med_ae, r2]).T
            df_report.columns = cols
            df_report.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                          '_simple_test_regression_report.csv'))

        return df_report



    def _train_eval_kloo(self, plot=False):
        # adapted from https://scikit-learn.org/


        if self.type_ == 'cls':
            if self.k_cv == 'k_fold':
                kf = KFold(n_splits=self.k)
                test_accs = []
                tprs = []
                aucs = []
                tprs_2 = []
                fprs = []
                precisions = []
                recalls = []
                mean_fpr = np.linspace(0, 1, 100)
                report = {}
                i = 0
                f = 0
                if self.n_categories <= 2:
                    fig, ax = plt.subplots(figsize=(14, 7))
                for train_index, test_index in kf.split(self.X):
                    f += 1
                    print('Training fold: ', f)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    preds = self.model.predict(X_test)
                    probs = self.model.predict_proba(X_test)
                    test_accs.append(100 * metrics.accuracy_score(y_test, preds))
                    rep = metrics.classification_report(y_test, preds, output_dict=True)
                    rep = self.unravel_report(rep)
                    report = self.combine_dicts(report, rep)

                    if self.n_categories <= 2:
                        fpr, tpr, _ = metrics.roc_curve(y_test, probs[:, 0])
                        tprs_2.append(tpr)
                        fprs.append(fpr)
                        precisions.append(metrics.precision_score(y_test, preds, average=None))
                        recalls.append(metrics.recall_score(y_test, preds, average=None))
                        viz = metrics.plot_roc_curve(self.model, X_test, y_test,
                                             name='ROC fold {}'.format(i),
                                             alpha=0.3, lw=1, ax=ax)
                        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        aucs.append(viz.roc_auc)
                    i += 1

                if self.n_categories <= 2:
                    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                            label='Chance', alpha=.8)

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = metrics.auc(mean_fpr, mean_tpr)
                    std_auc = np.std(aucs)
                    ax.plot(mean_fpr, mean_tpr, color='b',
                            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                            lw=2, alpha=.8)

                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                    label=r'$\pm$ 1 std. dev.')

                    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                           title="Receiver operating characteristic example")
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' +'_k_fold_AUROC.png'))
                    if plot:
                        plt.show()
                    plt.close()

                    recalls = np.asarray(recalls)
                    precisions = np.asarray(precisions)
                    tprs = np.asarray(tprs)
                    tprs_2 = np.asarray(tprs_2)
                    fprs = np.asarray(fprs)
                    aucs = np.asarray(aucs)

                test_accs = np.asarray(test_accs)
                if self.n_categories <= 2:
                    results = pd.DataFrame([test_accs, tprs, aucs, tprs_2, fprs, precisions, recalls]).T
                    cols = ['test_accs', 'tprs', 'aucs', 'tprs_2', 'fprs', 'precision', 'recall']
                    results.columns = cols
                    results = results.append(results.mean(), ignore_index=True)
                    results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                                '_k_fold_test_classification_report.csv'))
                else:
                    results = pd.DataFrame(report)
                    results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                                '_k_fold_test_classification_report.csv'))


            elif self.k_cv == 'loo_cv':
                kf = LeaveOneOut()
                y_probs = []
                y_preds = []
                y_GTs = []
                i = 0
                for train_index, test_index in kf.split(self.X):
                    i += 1
                    print('Loo-cv split number: ', i)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    pred = self.model.predict(X_test)
                    prob = self.model.predict_proba(X_test)
                    y_preds.append(pred)
                    y_probs.append(prob)
                    y_GTs.append(y_test.values[0])

                y_GTs = np.asarray(y_GTs)
                y_probs = np.asarray(y_probs)
                y_preds = np.asarray(y_preds)
                test_accs = 100 * metrics.accuracy_score(y_GTs, y_preds)
                conf_mat = np.asarray(metrics.confusion_matrix(y_GTs, y_preds))
                np.savetxt(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' + str(self.class_) + '_loocv_train_test_conf_mat.txt'), conf_mat)
                results = metrics.classification_report(y_GTs, y_preds, output_dict=True)
                results = pd.DataFrame(results).transpose()
                results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                              '_loocv_test_classification_report.csv'))


        elif self.type_ == 'reg':
            if self.k_cv == 'k_fold':
                kf = KFold(n_splits=self.k)
                expl_vars = []
                maes = []
                mses = []
                msles = []
                r2s = []
                med_aes = []
                f = 0
                for train_index, test_index in kf.split(self.X):
                    f += 1
                    print('Training fold: ', f)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    preds = self.model.predict(X_test)
                    expl_vars.append(metrics.explained_variance_score(y_test, preds))
                    maes.append(metrics.mean_absolute_error(y_test, preds))
                    mses.append(metrics.mean_squared_error(y_test, preds))
                    try:
                        msles.append(metrics.mean_squared_log_error(y_test, preds))
                    except:
                        pass
                    med_aes.append(metrics.median_absolute_error(y_test, preds))
                    r2s.append(metrics.r2_score(y_test, preds))

                expl_vars = np.asarray(expl_vars)
                maes = np.asarray(maes)
                mses = np.asarray(mses)
                msles = np.asarray(msles)
                r2s = np.asarray(r2s)
                med_aes = np.asarray(med_aes)

                results = pd.DataFrame([expl_vars, maes, mses, msles, med_aes, r2s]).T
                cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
                results.columns = cols
                results = results.append(results.mean(), ignore_index=True)
                print('Last row is the mean across columns.')
                results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                     '_k_fold_test_regression_report.csv'))


            elif self.k_cv == 'loo_cv':
                kf = LeaveOneOut()
                y_preds = []
                y_GTs = []
                i = 0
                for train_index, test_index in kf.split(self.X):
                    i += 1
                    print('Loo-cv split number: ', i)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    pred = self.model.predict(X_test)
                    y_preds.append(pred)
                    y_GTs.append(y_test.values[0])

                y_GTs = np.asarray(y_GTs)
                y_preds = np.asarray(y_preds)

                expl_var = metrics.explained_variance_score(y_GTs, y_preds)
                mae = metrics.mean_absolute_error(y_GTs, y_preds)
                mse = metrics.mean_squared_error(y_GTs, y_preds)
                msle = 0
                try:
                    msle = metrics.mean_squared_log_error(y_GTs, y_preds)
                except:
                    pass
                med_ae = metrics.median_absolute_error(y_GTs, y_preds)
                r2 = metrics.r2_score(y_GTs, y_preds)
                results = pd.DataFrame([expl_var, mae, mse, msle, med_ae, r2]).T
                cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
                results.columns = cols
                results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                            '_loocv_test_regression_report.csv'))

        return results


    def tune_model(self, tunable_params=None, folds=3, n_iter=100):
        '''
        :param tunable_params: list of desired parameters to tune over
        :param folds: number of folds for CV
        :param n_iter: number of iters for tuning
        :return: model with optimal hyperparameters
        '''
        print('Tuning the following parameters: ', tunable_params)

        assert self.k_cv == 'split',\
            'k_cv must be set to "split" to allow hyperparameter tuning with a designated training set!'

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

        filtered = {k: v for k, v in random_grid.items() if v is not None}
        random_grid.clear()
        random_grid.update(filtered)
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
        :return: explainer and shapley values
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
            joblib.dump(explainer, os.path.join(self.output_dir, self.outcome_var + '_linear_explainer.sav'))
            joblib.dump(shap_vals, os.path.join(self.output_dir, self.outcome_var + '_linear_explainer_shap_values.sav'))

        return explainer, shap_vals

    def shap_bootstrap(self, modell=None, retrain=False, n_bootstraps=1000, n_samples=100, class_ind=0):
        '''

        :param modell: desired model  (if None then will use model that you trained earlier within this class)
        :param retrain: this retrains the model for each bootstrap
        :param n_bootstraps: number of boostraps to undertake
        :param n_samples: sample size used in each bootstrap
        :param class_ind: for multiple classes, if you want to look at a particular class
        :return: shapley values for all bootstraps, and report for mean and standard error
        '''

        assert n_samples < len(self.X_test), 'n_samples cannot be longer than the test set! Reduce n_samples.'

        if modell == None:
            modell = self.model
        shap_vals_bootstraps = []

        if self.k_cv == 'split':
            X_test_bootstrap = self.X_test
        elif self.k_cv == 'loo_cv' or self.k_cv == 'k_fold':
            X_test_bootstrap = self.X
        indices = np.arange(0, len(X_test_bootstrap))
        b = 0
        for _ in range(n_bootstraps):
            b += 1
            if b % 20 == 0:
                print('Bootstrap number: ', b)
            if retrain:
                modell = self.make_model(self.config)
                modell, _ = self.train_test(plot=False)
            sample_inds = np.random.choice(indices, n_samples)
            X_test_sample = X_test_bootstrap.iloc[sample_inds]
            if self.class_ == 'RF':
                explainer = shap.TreeExplainer(model=modell, model_output='margin')
                shap_vals = explainer.shap_values(X_test_sample)

                if self.type_ == 'cls':
                    shap_vals = shap_vals[class_ind]

            elif self.class_ == 'lin':
                explainer = shap.LinearExplainer(modell, X_test_sample)
                shap_vals = explainer.shap_values(X_test_sample)

            shap_vals_bootstraps.append(shap_vals)

        shap_vals_bootstraps = np.asarray(shap_vals_bootstraps)
        abs_shaps = np.abs(shap_vals_bootstraps)
        mean_shaps = np.mean(abs_shaps, 1)
        se_bootstraps = sem(mean_shaps, 0)
        mean_bootstraps = np.mean(mean_shaps, 0)
        sorted_inds = np.argsort(mean_bootstraps)

        plt.figure(figsize=(7, 10))
        plt.barh(X_test_bootstrap.columns[sorted_inds][-20:], mean_bootstraps[sorted_inds][-20:], xerr=se_bootstraps[sorted_inds][-20:] * 1.96)
        plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, self.outcome_var + 'bootstrap_shap.png'))
        plt.show()

        bootstrap_results = pd.DataFrame([mean_bootstraps, se_bootstraps])
        bootstrap_results.columns = self.X_test.columns
        bootstrap_results.to_csv(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) +
                                                '_' + str(self.class_) + '_shapley_importances_bootstrap_results.csv'))
        return shap_vals_bootstraps, bootstrap_results


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
                plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                         str(self.class_) + 'shap_val_summary.png'))
                plt.show()
                plt.close()
            else:
                shap.summary_plot(shap_values=shap_vals[class_ind], features=self.X_test, max_display=20, plot_type='bar', show=False)
                plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                         str(self.class_) + 'shap_val_summary.png'))
                plt.show()
                plt.close()

            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals[class_ind], features=self.X_test, max_display=20, plot_type='dot',
                              show=False)
            else:
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='dot',
                              show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + '_' + str(class_ind) + '_shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                     shap_values=shap_vals[class_ind], features=self.X_test, show=False)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                             shap_values=shap_vals, features=self.X_test, show=False)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_'
                                         + str(self.class_) + 'shap_interaction_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()

        elif self.type_ == 'reg':

            shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='bar',
                              show=False)
            plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + 'shap_val_summary.png'))
            plt.show()
            plt.close()

            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='dot',
                                  show=False)
            else:
                shap.summary_plot(shap_values=shap_vals, features=self.X_test, max_display=20, plot_type='dot',
                          show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + 'shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals, features=self.X_test, show=False)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals, features=self.X_test, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                         str(self.class_) + 'shap_interaction_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()

    def unravel_report(self, report):
        new_d = {}
        for key in report.keys():
            p1 = report[key]
            if key == 'accuracy':
                new_d[key] = report[key]
            try:
                _ = float(key)
                for key2 in p1.keys():
                    new_key_name = str(key) + ' ' + str(key2)
                    new_d[new_key_name] = report[key][key2]

            except:
                try:
                    for key2 in p1.keys():
                        new_key_name = str(key) + ' ' + str(key2)
                        new_d[new_key_name] = report[key][key2]
                except:
                    pass
        return new_d

    def combine_dicts(self, d_all, d_add):

        for key in d_add.keys():
            if key in d_all.keys():
                d_all[key] = np.concatenate((d_all[key], np.array([d_add[key]])))

            else:
                d_all[key] = np.asarray([d_add[key]])
        return d_all







