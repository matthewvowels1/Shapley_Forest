# Shapley_Forest

RFShap is a Class designed to speed up the use of RF and linear regressors/classifiers with Shapley values,
hyperparameter tuning, and bootstrapping for confidence intervals. The expected userbase comprises psychologists and
social scientists. Some of the built in methods/functions provide outputs, but actually, these outputs are not
necessary to run all the features, they are just there for convenience. The Class is still being developed, so we
welcome suggestions. 

The ipynb is a jupyter notebook that walks through an example.


## 0. Requirements
pandas == 1.0.5

os

shap == 0.35.0

numpy == 1.18.5

sklearn == 0.23.1

scipy == 1.5.0

imblearn == 0.7.0

matplotlib == 3.2.2

joblib == 0.15.1


## 1.Create environment:

First, get RF_Shap.py into your current working directory. Although this isn't necessary, if you are relatively
new to python it will make it easier to import with:

```python from RF_Shap import RFShap```

Specify your output directory where all your results are going to go, and intialize the RFShap environment with some
settings. If `model_dir` is not None, the environment will automatically try to load a pre-trained model from that 
directory. BE CAREFUL HERE: if the model has been pretrained on a different train/test split, then you need to watch
out for overfitting in cases where train and test sets overlap. This is where `trn_tst_split` is useful in combination
with `random_seed`. Try to be consistent and careful how you set/change these..
(ideally, choose some values and stick to them). NOTE also that if you set `k_cv='loo_cv' or 'k_fold`, it will
override the tr/tst split and will not let you do hyperparameter tuning. Also if `k_cv = 'k_fold'`, k=... will 
allow you to set the number of folds.

```python
od = '/home/matthewvowels/GitHub/Psych_ML/Shapley_Forest/output'
shap_env = RFShap(model_dir=None, exclude_vars=['gay', 'lesbian'], outcome_var='age',
                  output_dir=od, random_seed=42, class_='RF',type_='reg', balanced='balanced', trn_tst_split=0.6,
                    k_cv='split', k=5)
```



The following models are used:
 
`class_ = 'RF', type_ = 'reg'` : sklearn.ensemble.RandomForestRegressor

`class_ = 'RF', type_ = 'cls'` : sklearn.ensemble.RandomForestClassifier

`class_ = 'RF', type_ = 'cls', balanced='balanced'` : imblearn.ensemble.BalancedRandomForestClassifier

`class_ = 'lin', type_ = 'cls', balanced='balanced'/None` : sklearn.linear_model.LogisticRegression

`class_ = 'lin', type_ = 'reg'` : sklearn.linear_model.LinearRegression

The balancing either calls imblearns RF, or enables class weight balancing for logistic regression.

## 2. Feed the Class some data.
Once the class has been initialized up, we can feed it a DataFrame. Note that the variable names you used when initializing
the environment (and some of the var names used below) need to appear in the dataset columns! 

Feed dataframe and process it:

``` python
dataset, X, y, xtr, xts, ytr, ytst  = shap_env.munch(dataset=dataset)
```

This will remove your `exclude_vars` and create the train/test split that should be used for all subsequent
 hyperparameter tuning and testing - so make sure you are happy with this split!

## 3. Create a model

To create a model, you have the option to feed a `config` dictionary, containing model settings.

If none are set, then defaults will be used. See the bottom of this README for other config options.



Create model:

```python

config = {'n_estimators': 100, 'criterion': 'mse', 'max_depth': None, 'min_samples_split': 2,
          'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto',
          'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
          'bootstrap': True, 'oob_score': True, 'n_jobs': None, 'random_state': None, 
          'verbose': 0, 'warm_start': False, 'max_samples': None}
model = shap_env.make_model(config)
```

If no output is required (i.e. remove `model = ` above) then the model will still be created and contained 
in the Class. But you might want the model for other nefarious purposes ; ) so you get the option to access it 

Another way to get it is via `model = shap_env.model`.


## 4. Do a quick train/test pass using your supplied config (or defaults):

```python
shap_env.train_test(plot=False)
```

If we have `k_cv = 'k_fold'` or `'loo_cv'` then they will be used here. `plot` argument refers to k-fold training and the
creation of a per-fold auroc plotting feature.

TODO: make plotting feature conditional on whether outcome is binary or categorical.


### Save/load models:
At this point, you could save your model using:

`shap_env.save_model()`

This will use the model_dir specified at class init. However, you can also specify a new dir e.g.:

`shap_env.save_model('desktop/dir/model.sav')`

You can load this model back into the class in various ways.

If you have already specified a model_dir:

`model = shap_env.load_model()`

or if you haven't:

`model = shap_env.load_model('desktop/dir/model.sav)`

where you don't necessarily need to precede the function call with `model = `
as the model will be contained in the class anyway.


## 5. HyperParameter Tuning

Do hyperparameter tuning by first specifying a list of parameters to tune. The available tunable params
depend on the model type and class as follows:


```python
tunable_params = ['n_estimators', 'max_depth', 'max_features']
tuned_model = shap_env.tune_model(tunable_params=tunable_params, folds=5, n_iter=100)
```

NOTE that tuned_model has not been trained yet! It just has tuned hyperparameters.

Options for tuning different model types:
```python
tunable_params_rf = ['n_estimators', 'criterion', 'max_features', 'max_leaf_nodes', 'max_depth',
                               'min_samples_leaf', 'min_samples_split', 'bootstrap']
tunable_params_linreg = ['normalize', 'fit_intercept']
tunable_params_logreg = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'solver', 'max_iter']
```

If you specify a hyperparameter that cannot be trained, an error will be flagged and it will be ignored.


## 6. Training the tuned model and running Shapley stuff

Train model with above hyperparams, and run shapley stuff:

```python
tuned_model, report = shap_env.train_test()
explainer, shap_vals = shap_env.run_shap_explainer(modell=tuned_model)
shap_env.shap_plot(shap_vals=shap_vals, specific_var='mtf', interaction_var='ftm', classwise=True, class_ind=1)
```

`classwise` is for the importances plot when using RFs - it splits the importances for each class. 
If not using RFs, it is ignored. For the specific feature plots, we can see how these impact the output of the model,
 and also include interaction variables. Class_ind is for when we want to plot according to a specific class
  (e.g. {0,1} for binary).
This method outputs images and also saves the explainer and shap_vals.

## 7. Bootstrapping feature importances

The class includes a function to calculate shap_vals for samples (with replacement). This builds up
95 percent confidence intervals around the importance values.

`shap_vals_bootstrap, results = shap_env.shap_bootstrap(modell=model, retrain=False, n_bootstraps=1000, n_samples=50, class_ind=0)`

Here retrain allows you to specify whether you want to retrain your model for every bootstrap. Obviously this is very
computationally expensive, particularly when you have `shap_env.k_cv = 'loo_cv'` so it is recommended, if you DO 
want to retrain, to have `shap_env.k_cv = 'split'`

The output provides means and standard errors, as well as the full list of shap_vals over all bootstraps.










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



## References:

Code has been adapted from the following sources (please check them out!):

https://deep-and-shallow.com/2019/11/24/interpretability-cracking-open-the-black-box-part-iii/

https://github.com/manujosephv/interpretability_blog/blob/master/census_income_interpretability.ipynb

https://github.com/slundberg/shap
