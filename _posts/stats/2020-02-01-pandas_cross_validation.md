---
layout: post
category: blog
title:  "Modularization: Saving Model Predictions under Cross-Validation"
date:   2020-02-01
tags:
  - infrastructure
  - pandas
  - numpy
  - pipeline
  - parallelization
  - optimization
---


*Tutorials rarely discuss how to save and process intermediary results in more complex experimental setting. This is a walkthrough of how to save model predictions to shift post-processing and evaluation into a separate step. We make the model predictions in a setting doing cross-validation while parallelizing model training using {multiprocessing} and save the predictions using {numpy}.*


```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
```


```python
model_lib = {}
pred_train = {}
pred_test = {}
eval_test = {}
```


```python
X, y = make_classification(n_samples=1000)
```

These are the models that we will use as examples. In a real experiment, these will have been tuned beforehand or will be a model pipeline including cross validation.


```python
model_lib['logit'] = LogisticRegression(C=1, solver="lbfgs")
model_lib['random_forest'] = RandomForestClassifier(n_estimators=100, min_samples_leaf=50)
```

## Simple train test split

The simple case discussed in most introductory texts is a single train-test split, where we have one training set and one test set. 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Fit all the models


```python
for model in model_lib.values():
    model.fit(X_train, y_train)
```

Make predictions on train and test data, so we could save these for further analysis or reproducability. 


```python
for model_name, model in model_lib.items():
    pred_train[model_name] = model.predict_proba(X_train)[:,1]
    pred_test[model_name] = model.predict_proba(X_test)[:,1]
```

Evaluate the predictions of each model for which we have saved test predictions


```python
for model_name, model in pred_test.items():
    eval_test[model_name] = {'ROC_AUC':roc_auc_score(y_test, pred_test[model_name])}
```


```python
eval_test
```




    {'logit': {'ROC_AUC': 0.9651639344262295},
     'random_forest': {'ROC_AUC': 0.9577356557377049}}




```python
pd.DataFrame(eval_test)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logit</th>
      <th>random_forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC_AUC</th>
      <td>0.965164</td>
      <td>0.957736</td>
    </tr>
  </tbody>
</table>
</div>



Clean the workspace for the next part


```python
del(pred_train, pred_test, eval_test)
```

## Cross validation

We often want more than a single test result to check the robustness of our estimate. A good way to do that is to do cross validation for evaluation of the final, tuned model(s). The issue is that cross-validation takes more time, so we would like to A) parallelize training on the folds and B) to save the model predictions so that we can redo our evaluation if we need to without repeating model training. 


```python
import multiprocessing as mp
```

Get the cross-validation indices.


```python
np.random.seed(123456789)
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=123456789)
folds = list(splitter.split(X, y))
```

Fit the model fitting steps above into a function, so that we can easily parallelize it. 


```python
def model_pipeline(model_lib, X, y, train_test_indices):
    model_lib = model_lib.copy()
    pred_train = {}
    pred_test = {}
    
    X_train, y_train, X_test, y_test = X[train_test_indices[0],:], y[train_test_indices[0]], X[train_test_indices[1],:], y[train_test_indices[1]]
    
    # Fit the models
    for model in model_lib.values():
        model.fit(X_train, y_train)
    
    # Make a prediction
    for model_name, model in model_lib.items():
        pred_train[model_name] = model.predict_proba(X_train)[:,1]
        pred_test[model_name] = model.predict_proba(X_test)[:,1]
    
    # Return only the model predictions, but for the train and test set
    return({'train':(train_test_indices[0], pred_train), 'test':(train_test_indices[1],pred_test)})
    
```

For parallelization with a pool in {multiprocessing}, we will need a callback that combines the results on each fold. We save the results for each fold in a list, one for the training data and one for the test data. I prefer to separate them, because we will work with only the test predictions for the evaluation, but may want to keep the train set prediction to check for overfitting or sanity checks. 


```python
def log_result(x):
    try:
        pred_train.append(x['train'])
        pred_test.append(x['test'])
    except:
        pred_train.append(None)
        pred_test.append(None)

pred_train = []
pred_test = []
    
```

In the next step, we call the model training function on each fold in parallel, wait for the functions to complete and then close the pool.


```python
# Open the pool
pool = mp.Pool(3)

# Apply training steps on each fold in parallel
for i,fold in enumerate(folds):
    pool.apply_async(model_pipeline, 
        args=(model_lib, X, y, fold), 
        callback = log_result)

# Close the pool
pool.close()
pool.join()
print("Cross-Validation complete.")
```

    Cross-Validation complete.


These are the raw predictions that we usually want to save. Don't forget to also save the correct version of the input data. With the data, the train and test indices and the predictions we can repeat or alter the evaluation without having to retrain the model library. 

We are discarding the actual models here, which you may want to pickle to look into model interpretation later. The best way to save the model predictions is to use numpy. We could also do a json.dump for a more flexible format, but then the numpy arrays would have to be preprocessed, since {json} doesn't handle them out-of-the-box.


```python
np.save( "./model_predictions_test", pred_test, allow_pickle=True)
del(pred_test)
```


```python
# Re-load the data this way
pred_test = np.load( "./model_predictions_test.npy", allow_pickle=True)
```

Wrap the steps above up into a function that evaluates several predictions and returns a dictionary of results for each model (1st level) and metric (2nd level)


```python
def eval_from_dict(y_true, prediction_dict):
    output = {}
    for model_name, pred in prediction_dict.items():
        output[model_name] = {'ROC_AUC':roc_auc_score(y_true, pred)}
    return output


eval_test = [eval_from_dict(y[idx], pred_dict) for idx, pred_dict in pred_test]
```


```python
eval_test
```




    [{'logit': {'ROC_AUC': 0.9650399799203987},
      'random_forest': {'ROC_AUC': 0.9641435691491269}},
     {'logit': {'ROC_AUC': 0.970460153868486},
      'random_forest': {'ROC_AUC': 0.9683190593700102}},
     {'logit': {'ROC_AUC': 0.9552511742981104},
      'random_forest': {'ROC_AUC': 0.9597690845853203}}]



Merge the results for each fold into a dataframe with hierarchical index 1. folds 2. metrics and the models in the columns. Inputting the keys creates the multi-index for the three folds, with the metric(s) as indices at the level below. 


```python
eval_test_dataframe = pd.concat([pd.DataFrame(x) for x in eval_test], axis=0, keys=range(len(eval_test)))
```


```python
eval_test_dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>logit</th>
      <th>random_forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>ROC_AUC</th>
      <td>0.965040</td>
      <td>0.964144</td>
    </tr>
    <tr>
      <th>1</th>
      <th>ROC_AUC</th>
      <td>0.970460</td>
      <td>0.968319</td>
    </tr>
    <tr>
      <th>2</th>
      <th>ROC_AUC</th>
      <td>0.955251</td>
      <td>0.959769</td>
    </tr>
  </tbody>
</table>
</div>




```python
eval_test_dataframe.index.rename(["fold","metric"], inplace=True)
```


```python
eval_test_dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>logit</th>
      <th>random_forest</th>
    </tr>
    <tr>
      <th>fold</th>
      <th>metric</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>ROC_AUC</th>
      <td>0.965040</td>
      <td>0.964144</td>
    </tr>
    <tr>
      <th>1</th>
      <th>ROC_AUC</th>
      <td>0.970460</td>
      <td>0.968319</td>
    </tr>
    <tr>
      <th>2</th>
      <th>ROC_AUC</th>
      <td>0.955251</td>
      <td>0.959769</td>
    </tr>
  </tbody>
</table>
</div>



Average the results over the folds easily using panda's groupby.


```python
eval_test_dataframe.groupby("metric").mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logit</th>
      <th>random_forest</th>
    </tr>
    <tr>
      <th>metric</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC_AUC</th>
      <td>0.963584</td>
      <td>0.964077</td>
    </tr>
  </tbody>
</table>
</div>




```python
eval_test_dataframe.groupby("metric").std()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logit</th>
      <th>random_forest</th>
    </tr>
    <tr>
      <th>metric</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC_AUC</th>
      <td>0.007708</td>
      <td>0.004275</td>
    </tr>
  </tbody>
</table>
</div>


