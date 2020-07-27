# Shapley_Forest

Create environment:

```python
od = '/home/matthewvowels/GitHub/Psych_ML/Shapley_Forest/output'
shap_env = RFShap(model_dir=None, exclude_vars=['gay', 'lesbian'], outcome_var='age',
                  output_dir=od, random_seed=42, class_='RF',type_='reg', balanced='balanced', trn_tst_split=0.6)
```

Feed dataframe and process it:

``` python
dataset, X, y, xtr, xts, ytr, ytst  = shap_env.munch(dataset=dataset)
```

Create model:

```python

config = {'n_estimators': 100, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 2,
          'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto',
          'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
          'bootstrap': True, 'oob_score': True, 'n_jobs': None, 'random_state': None, 
          'verbose': 0, 'warm_start': False, 'max_samples': None}
model = shap_env.make_model(config)
```

here, `class_` is 'RF' or 'lin' for random forest and linear/logistic models;
`type_` is 'reg' or 'cls' for regression or classification; and `balanced` is 'balanced' or 'unbalanced' to enable the use of imblearn's RF classifier (not applicable for regression models).


Quickly train and test the model with a config:

```python
shap_env.train_test()
```


Do hyperparameter tuning:


```python
tunable_params = ['n_estimators', 'max_depth', 'max_features']
tuned_model = shap_env.tune_model(tunable_params=tunable_params, folds=5, n_iter=100)
```

Options for tuning different model types:
```python
tunable_params_rf = ['n_estimators', 'max_depth', 'max_features']
tunable_params_linreg = ['normalize', 'fit_intercept']
tunable_params_logreg = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'solver', 'max_iter']
```

Train model with above hyperparams, and run shapley stuff:

```python
tuned_model, report = shap_env.train_test()
tuned_model, report = shap_env.train_test()
explainer, shap_vals = shap_env.run_shap_explainer(modell=tuned_model)
shap_env.shap_plot(shap_vals=shap_vals, specific_var='mtf', interaction_var='ftm', classwise=True, class_ind=1)
```

`classwise` is for the importances plot when using RFs - it splits the importances for each class. If not using RFs, it is ignored. For the specific feature plots, we can see how these impact the output of the model, and also include interaction variables. Class_ind is for when we want to plot according to a specific class (e.g. {0,1} for binary).










### Possible Hyperparameters are:
REGRESSOR RF
``` python
config = {'n_estimators': 100, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 2,
          'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto',
          'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
          'bootstrap': True, 'oob_score': True, 'n_jobs': None, 'random_state': None, 
          'verbose': 0, 'warm_start': False, 'max_samples': None}
```
CLASSIFIER RF
```python
config_cls = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
          'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto',
          'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 
          'bootstrap': True, 'oob_score': True, 'n_jobs': None, 'random_state': None, 
          'verbose': 0, 'warm_start': False, 'max_samples': None}
```

LINEAR REGRESSION
```python
config_linreg = {'fit_intercept': True, 'normalize': False, 'n_jobs': None}
```

LOGISTIC REGRESSION
```python
config_logreg = {'penalty': 'l2', 'dual': False, 'tol': 1e-4, 'C': 1.0, 'fit_intercept': True,
                'intercept_scaling': 1.0, 'solver': 'lbfgs', 'class_weight': None, 'max_iter': 100, 'multi_class': 'auto',
                'verbose': 0, 'n_jobs': None}
```





