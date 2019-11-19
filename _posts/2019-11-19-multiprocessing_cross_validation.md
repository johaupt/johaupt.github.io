---
layout: post
title:  "Parallelizing Model Selection Using the Multiprocessing Library in Python"
date:   2019-11-19
categories:
  - python
  - parallel processing
  - cross-validation
---


*A quick guide on using Python's `multiprocessing` library to parallelize model selection using `apply_async`*

## The problem

Some common data science tasks take a long time to run, but are embarrassingly parallel. Embarrassingly parallel means that they do not depend on each other and could therefore easily be done at the same time. The best examples are training different models and cross validation. In cross validation, training the model on *k-1* folds before testing it on the remaining fold and training the model on *k-1* different folds before testing it on a different remaining fold are two tasks that are not connected. Because they are not connected, we can handle them to different workers and process them in parallel. 

Scikit-learn has parallization implemented using its `n_jobs` option, but we don't need to rely on its ecosystem to parallelize model selection. Instead, we will use the `multiprocessing` library directly.


```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
```


```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=123)
```

We can use `scikit-learn` to conveniently generate the indices for the training and test data for a number of cross-validation folds


```python
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
folds = list(splitter.split(X, y))
```


```python
len(folds)
```




    3




```python
len(folds[0])
```




    2



## A cross validation helper function


```python
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
```


```python
def benchmark_models(X, y, split):
    """
    Helper function to benchmark models
    X : array
    y : array
    split : tuple
     Training and test indices (train_idx, test_idx)
    """
    X_train, y_train = X[split[0],:], y[split[0]]
    X_test, y_test   = X[split[1],:], y[split[1]]
    
    
    model_library = {}
    # One candidate model
    model_library["logit"] = LogisticRegression(solver='liblinear')
    # Another candidate model
    model_library["rf"] = RandomForestClassifier(n_estimators=100, min_samples_leaf=20)

    results = {}
    for model_name, model in model_library.items():
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions on the test data
        pred_test = model.predict_proba(X_test)[:,1]
        # Evaluate the model
        results[model_name] = roc_auc_score(y_test, pred_test)
    
    return pd.DataFrame(results, index = ["ROC-AUC"])
        
```

Test the function


```python
benchmark_models(X,y,split=folds[0])
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
      <th>rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC-AUC</th>
      <td>0.865646</td>
      <td>0.88949</td>
    </tr>
  </tbody>
</table>
</div>



## The multiprocessing library

The `multiprocessing` library is not particularly geared towards statistics, like scikit-learn. It was difficult for me to figure out which of its functionality is useful for our problem of parallelizing cross validation. To sum up our requirements, we want to:
- Run multiple tasks in parallel
- Pass several arguments to the function


```python
import multiprocessing as mp
```

We first specify how many processes we want to run in parallel. This number is restricted by the number of cores available. I like to use all but one core on my machine (to avoid programs freezing). On a shared computation server, make sure you understand the policy and be polite by leaving resources for others.


```python
#pool = mp.Pool(3)
# Python can count the available cores for you in most cases
pool = mp.Pool(mp.cpu_count()-1)
```

Function `apply_async` can be used to send function calls including additional arguments to one of the processes in the pool. In contrast to `apply` (without the `_async`), the program will not wait for each call to be completed before moving on. We can therefore assign the first cross-validation iteration and immidiately assign the second iteration before the first iteration is completed.  
There is a drawback to `apply_async` in that it does not return the result after the call complete. Instead, it returns another object with a `get()` method. A more convenient solution is a callback. The callback function will be called on the result once the function call is completed. So we'll specify a list for the results and a callback to save each result into that list. 


```python
results = []
def log_result(x):
    results.append(x)
```

An important intuition with `apply` and `apply_async` is that we assign a single function call to a worker when we call the function. In contrast, the `map` functionality would assign a list of tasks to available workers at once. `apply_async` calls the workers into your office one by one to explain their task to them.  
IMPORTANT: The results will not come back in the same order as we assigned the tasks. If we want to match the results to each fold, then we should pass an identifier to the function. 


```python
for fold in folds:
    pool.apply_async(benchmark_models, args=(X, y, fold), callback = log_result)
```

After assigning each task the program moves on without waiting for the result from the worker. That was convenient when we assigned the tasks and didn't want to wait for the first result before assigning the second task. But we ususally want to wait for all results before moving on with script and, for example, average the results.  
We tell the program to wait for all workers to complete their tasks using the method `join()`. Before we do so, we are required to make sure that no new tasks are assigned, which we do by using `close()` on the pool. 


```python
# Close the pool for new tasks
pool.close()
# Wait for all tasks to complete at this point
pool.join()
```

After collecting the results, we can work with the data as usual. 


```python
result = pd.concat(results, axis=0, sort=True)
```


```python
result
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
      <th>rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC-AUC</th>
      <td>0.843773</td>
      <td>0.911090</td>
    </tr>
    <tr>
      <th>ROC-AUC</th>
      <td>0.865646</td>
      <td>0.885977</td>
    </tr>
    <tr>
      <th>ROC-AUC</th>
      <td>0.893829</td>
      <td>0.909498</td>
    </tr>
  </tbody>
</table>
</div>



For cross validation, we would usually average the results over all splits and then compare our models.


```python
result.index.name = "metric"
result.reset_index()
average = result.groupby(['metric']).mean()
```


```python
average
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
      <th>rf</th>
    </tr>
    <tr>
      <th>metric</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ROC-AUC</th>
      <td>0.867749</td>
      <td>0.902188</td>
    </tr>
  </tbody>
</table>
</div>


