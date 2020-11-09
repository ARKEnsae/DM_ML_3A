**NAMES :  Kim ANTUNEZ, Isabelle BERNARD (Group : Mr Denis)**

<h1><center> TP1 : Basic functions for Supervised Machine Learning. </center></h1>

The deadline for report submission is Tuesday, November 10th 2020.

Note: the goal of this first TP is to become familiar with 'sklearn' class in Python. In particular, we introduce most popular supervised learning algorithms. 

PART 1 is a list of commands that should be followed step by step. PART 2 is an open problem for which we are waiting for your creativity!

## Imported packages


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, confusion_matrix


%matplotlib notebook
```


```python
import random
random.seed(1) #to fix random and have the same results for both of us 
```

#  PART 1 -- MNIST


In the first part of TP1 we pursue the following goals:
1. Apply standard ML algorithms on a standard benchmark data
2. Learn basic means of data visualizations
3. Get familiar with sklearn's GridSearchCV and Pipeline

## Loading the data

MNIST dataset consists of black and white images of hand-written digits from $0$ to $9$ of size $28 \times 28$.
In this exercise we will work with a small from the original MNIST dataset. 

If you are interested in the whole dataset, execute the following commands
```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
```

Hence, the observations $(X_1, Y_1), \ldots, (X_n, Y_n)$ are such that $X_i \in \mathbb{R}^{784}$ and $Y_i \in \{0, \ldots, 9\}$. To be more precise, each component of vector $X_i$ is a number between $0$ and $255$, which signifies the intensity of black color.

The initial goal is to build a classifier $\hat g$, which receives a new image $X$ and outputs the number that is present on the image.


```python
X_train = np.load('data/mnist1_features_train.npy', allow_pickle=True)
y_train = np.load('data/mnist1_labels_train.npy', allow_pickle=True)
X_test = np.load('data/mnist1_features_test.npy', allow_pickle=True)
y_test = np.load('data/mnist1_labels_test.npy', allow_pickle=True)

n_samples, n_features = X_train.shape # extract dimensions of the design matrix
print('Train data contains: {} samples of dimension {}'.format(n_samples, n_features))
print('Test data contains: {} samples'.format(X_test.shape[0]))
```

    Train data contains: 2000 samples of dimension 784
    Test data contains: 200 samples
    

## Looking at the data

Since each observation is actually an image, we can visualize it.


```python
axes = plt.subplots(1, 10)[1]  # creates a grid of 10 plots

# More details about zip() function here 
# https://docs.python.org/3.3/library/functions.html#zip
images_and_labels = list(zip(X_train, y_train)) 
for ax, (image, label) in zip(axes, images_and_labels[:10]):
    ax.set_axis_off()
    ax.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('{}'.format(label))
```


![png](TP1_KA_IB_files/TP1_KA_IB_11_0.png)



```python
for i in range(10):
    print('Number of {}s in the train dataset is {}'.format(i, np.sum([y_train == str(i)])))
```

    Number of 0s in the train dataset is 196
    Number of 1s in the train dataset is 226
    Number of 2s in the train dataset is 214
    Number of 3s in the train dataset is 211
    Number of 4s in the train dataset is 187
    Number of 5s in the train dataset is 179
    Number of 6s in the train dataset is 175
    Number of 7s in the train dataset is 225
    Number of 8s in the train dataset is 186
    Number of 9s in the train dataset is 201
    

From the above we conclude that the dataset is rather balanced, that is, each class contains similar amount of observations. The rarest class is $y = 6$ with $175$ examples and the most common class is $y = 1$ with $226$ examples

## Cross-validation with GridSearchCV


**Question:** Explain in your report what happens when we run 
```python
clf.fit(X_train, y_train)
```
What is the complexity for each of the three following cases? 


**Answer :**  

The general objective here is to obtain a first classifier with the **KNN method**. To do that, we test different parameters of the KNN methods and choose the bests using a **cross validation**. That is to say that we test the KNN method by varying the number of neighbors from 1 to 5. The cross validation method used is called **the 3-fold Cross Validation** (CV) following those different steps:

1. we divide our training sample into 3 training sub-samples
2. we train the model on 2 samples and test it on the third one
3. We choose the parameter which has the best average test accuracy (see definition later) on the 3 samples.


**clf.fit(X_train, y_train)** applies what is described above to the training sample. It fits the model (learns from it) using X_train as training data and y_train as target values. The first clf (for classifier) used here is "KNeighborsClassifier" that is to say the k-nearest neighbors vote.


Let's imagine that you train a model on n points and it takes x minutes. If you train it on kn points, it takes kx minutes if the training time is linear, but sometimes it is more. For example, if it takes k2x, the training time is quadratic in the number of points. That is what we call the **complexity** of an algorithm. The question here is very broad (not very precise), because there are at least two different kinds of complexities : **training complexity and prediction complexity**. 

We define the complexity using a Big-O measure. It provides us with an asymptotic upper bound for the growth rate of the runtime of the chosen algorithm. Calling n the number of training samples and p the number of features the complexity predicted for the three methods are : 

* For the **knn** classifier : The parameter used for the algorithm is here ‘auto’. It selects ‘kd_tree’ if $k < N/2$ and the ‘effective_metric_’ is in the ‘VALID_METRICS’ list of ‘kd_tree’. It selects ‘ball_tree’ if k < N/2 and the ‘effective_metric_’ is not in the ‘VALID_METRICS’ list of ‘kd_tree’. It selects ‘brute’ if k >= N/2. For the brute-force method there is no training, (training complexity = $O(1)$),  but classifying has a high cost ($O(knp)$). kd-tree and ball_tree are $O(pnlog(n))$ for training and $O(klog(n))$ for prediction. 
. See precisions [here](https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5) and [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). 


* Support Vector Machines (**SVM**) are powerful tools, but their compute and storage requirements increase rapidly with the number of training vectors. The core of an SVM is a quadratic programming problem (QP), separating support vectors from the rest of the training data. The QP solver used by the libsvm-based implementation (`SVC` function) scales between O($pn^2$) and O($pn^3$) depending on how efficiently the libsvm cache is used in practice (dataset dependent). But recent approaches like [this one](https://www.cs.huji.ac.il/~shais/papers/SSSICML08.pdf) are inverse in the size of the training set. In the case of the `LinearSVC` method used in this "TP", it is indicated in the [documentation](https://scikit-learn.org/stable/modules/svm.html#complexity) that the implementation is much more efficient than its libsvm-based `SVC` counterpart and it's training complexity is $O(pn)$ and prediction one remains $O(n_{sv}*p)$ with $n_{sv}$ the number of Support Vectors.


* For **logistic regressions**, training complexity is $O(np)$ and prediction one is $O(p)$. See proof [here](https://levelup.gitconnected.com/train-test-complexity-and-space-complexity-of-logistic-regression-2cb3de762054).

*Main sources : [here](https://medium.com/@paritoshkumar_5426/time-complexity-of-ml-models-4ec39fad2770) and [here](https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/)*.


```python
# GridSearchCV with kNN : a simple baseline
knn = KNeighborsClassifier() # defining classifier
parameters = {'n_neighbors': [1, 2, 3, 4, 5]} # defining parameter space
```


```python
clf = GridSearchCV(knn, parameters, cv=3) #cross-validation : method 3-fold.
```


```python
clf.fit(X_train, y_train) #
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                                metric='minkowski',
                                                metric_params=None, n_jobs=None,
                                                n_neighbors=5, p=2,
                                                weights='uniform'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'n_neighbors': [1, 2, 3, 4, 5]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print('Returned hyperparameter: {}'.format(clf.best_params_))
print('Best classification accuracy in train is: {}'.format(clf.best_score_))
print('Classification accuracy on test is: {}'.format(clf.score(X_test, y_test)))
```

    Returned hyperparameter: {'n_neighbors': 1}
    Best classification accuracy in train is: 0.891497944721333
    Classification accuracy on test is: 0.875
    


```python
print(confusion_matrix(y_test, clf.predict(X_test)))
```

    [[21  0  0  0  0  0  1  0  0  0]
     [ 0 26  0  0  0  0  0  0  0  0]
     [ 0  0 14  0  0  2  0  0  0  0]
     [ 0  0  0 19  0  2  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 0  0  0  0  1  7  1  0  1  0]
     [ 0  0  0  0  0  1 23  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  1  0  1  0  0  0  0 14  1]
     [ 1  1  0  0  2  0  0  3  0 19]]
    

**Question:** What is the test accuracy? What would be the accuracy of random guess?

**Answer :**

Accuracy is the number of correctly predicted data points out of all the data points. The `accuracy_score` function computes the accuracy, either the fraction (default) or the count (`normalize=False`) of correct predictions.

More formally, for a **binary problem** it is defined as the number of true positives (y predicted 1 and with a true value of 1) and true negatives (y predicted 0 and with a true value of 0) divided by the number of true positives, true negatives, false positives (y predicted 1 and with a true value of 0), and false negatives (y predicted 0 and with a true value of 1). 

$$
\texttt{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1; otherwise it is 0.

If $\hat{y}_i$ is the predicted value of the $i$-th sample and $y_i$ is the corresponding true value, then the fraction of correct predictions over is defined as $n_\text{samples}$. Here is the normalized accuracy score : 

$$
\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)
$$

See more details [here](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) 

The accuracy is used to to determine which model is best at identifying relationships and patterns between variables in a dataset based on the input data or training data. Here we comment the "test accuracy" that is to say the accuracy of the test sample and not on the training sample !

The test accuracy  is here **0.875**. On a **random guess it would be 0.1** : one chance to be guess the right number out of 10 possible numbers (10 classes).



```python
#Simple example of accuracy for a multiclass analysis to illustrate.
y_true = [0, 0, 1, 2, 3]
y_pred = [0, 1, 2, 1, 3]
print("accuracy normalisée : ", accuracy_score(y_true, y_pred))
print("accuracy non normalisée : ", accuracy_score(y_true, y_pred, normalize=False))
```

    accuracy normalisée :  0.4
    accuracy non normalisée :  2
    

Indeed, 

$$
\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i) = \frac{1}{5} \sum_{i=0}^{4} 1(\hat{y}_i = y_i) =  \frac{1}{5} * 2 = 0.4
$$



**Question:** What is ``` LinearSVC()``` classifier? Which kernel are we using? What is ```C```? (this is a tricky question, try to find the answer online)

**Answer :** 

``` LinearSVC()``` (Linear Support Vector Classification) is a fast implementation of Support Vector Machine Classification (SVM) for the case of a linear kernel. It is similar to `SVC` with parameter `kernel=’linear’`, but implemented in terms of `liblinear` rather than `libsvm`, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

The main characteristics of this method are : 
- the loss used is the [‘squared_hinge’](https://en.wikipedia.org/wiki/Hinge_loss) (even if it is not indicated in the [general documentation](https://scikit-learn.org/stable/modules/svm.html#linearsvc) which is strange)
- to generate the multiclass problem, ` LinearSVC()` uses `  One-vs-All` (see example [here](http://eric.univ-lyon2.fr/~ricco/cours/slides/svm.pdf), slide 38).

More precisely, given training vectors $x_i \in \mathbb{R}^p$, i=1,…, n, in two classes, and a vector $y \in \{1, -1\}^n$ 
, our goal is to find $w \in \mathbb{R}^p$ and $b \in \mathbb{R}$ such that the prediction given by $\text{sign} (w^T\phi(x) + b)$ is correct for most samples.
LinearSVC solves the following problem:
$\min_ {w, b} \frac{1}{2} w^T w + C \sum_{i=1}\max(0, y_i (w^T \phi(x_i) + b))$,
where we make use of the hinge loss. This is the form that is directly optimized by LinearSVC, but unlike the dual form, this one does not involve inner products between samples, so the famous kernel trick cannot be applied. This is why only the linear kernel is supported by LinearSVC ($\phi$ is the identity function).

The **C parameter** is a regularization or penalty parameter. SVM only work properly if the data is separable. Otherwise, we will penalize the loss of this non-separability (see [here](https://scikit-learn.org/stable/modules/svm.html#svc)) measuring the distance between the misclassified points and the separating hyperplane. C represents misclassification or error term. The misclassification or error term tells the SVM optimisation how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. 
Concretely, when C is high, we penalize a lot for misclassification, which means that we classify lots of points correctly, also there is a chance to overfit.

Documentation : C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it: decreasing C corresponds to more regularization. LinearSVC and LinearSVR are less sensitive to C when it becomes large, and prediction results stop improving after a certain threshold. Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer.


**Question:** What is the outcome of ```np.logspace(-8, 8, 17, base=2)```? More generally, what is the outcome of ```np.logspace(-a, b, k, base=m)```?

**Answer :** 
```np.logspace(-8, 8, 17, base=2)``` returns 17 numbers spaced evenly on a log scale. The sequence starts at $2^{-8}$ and ends with $2^{8}$.

```np.logspace(-a, b, k, base=m)``` returns k numbers spaced evenly on a log scale (endpoint=True by default). The parameter `base` is the logarithmic base. In linear space, the sequence starts at $m^{-a}$ and ends at $m^b$. 

It is equivalent to
1. divide the interval $[-a,b]$ into $(y_i)_{i=1..k}$ $k$ equidistant points
2. return $\left(m^{y_i}\right)_{i=1..k}$


```python
# SVM Classifier
svc = LinearSVC(max_iter=5000)
parameters2 = {'C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
clf2 = GridSearchCV(svc, parameters2, cv=3)
clf2.fit(X_train, y_train)
```

    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    




    GridSearchCV(cv=3, error_score=nan,
                 estimator=LinearSVC(C=1.0, class_weight=None, dual=True,
                                     fit_intercept=True, intercept_scaling=1,
                                     loss='squared_hinge', max_iter=5000,
                                     multi_class='ovr', penalty='l2',
                                     random_state=None, tol=0.0001, verbose=0),
                 iid='deprecated', n_jobs=None,
                 param_grid={'C': array([3.90625e-03, 7.81250e-03, 1.56250e-02, 3.12500e-02, 6.25000e-02,
           1.25000e-01, 2.50000e-01, 5.00000e-01, 1.00000e+00, 2.00000e+00,
           4.00000e+00, 8.00000e+00, 1.60000e+01, 3.20000e+01, 6.40000e+01,
           1.28000e+02, 2.56000e+02])},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print('Returned hyperparameter: {}'.format(clf2.best_params_))
print('Best classification accuracy in train is: {}'.format(clf2.best_score_))
print('Classification accuracy on test is: {}'.format(clf2.score(X_test, y_test)))
```

    Returned hyperparameter: {'C': 0.00390625}
    Best classification accuracy in train is: 0.8095074084579332
    Classification accuracy on test is: 0.795
    

**Question** What is the meaning of the warnings? What is the parameter responsible for its appearence?

**Answer**

Warnings are about the fact that the algorithm does not converge considering the maximum number of iterations given. The maximum number of iterations is given by the parameter `max_iter` which is here set to `max_iter=5000`.

In fact, there is no preprocessing (date are not normalize/standardized data). Therefore unscaled data can slow down or even prevent the convergence of many metric-based and gradient-based estimators. Indeed many estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales.


```python
# SVM Classifier + Pipeline
pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', svc)])
parameters3 = {'svc__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
clf3 = GridSearchCV(pipe, parameters3, cv=3)
clf3.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler', MaxAbsScaler(copy=True)),
                                           ('svc',
                                            LinearSVC(C=1.0, class_weight=None,
                                                      dual=True, fit_intercept=True,
                                                      intercept_scaling=1,
                                                      loss='squared_hinge',
                                                      max_iter=5000,
                                                      multi_class='ovr',
                                                      penalty='l2',
                                                      random_state=None, tol=0.0001,
                                                      verbose=0))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'svc__C': array([3.90625e-03, 7.81250e-03, 1.56250e-02, 3.12500e-02, 6.25000e-02,
           1.25000e-01, 2.50000e-01, 5.00000e-01, 1.00000e+00, 2.00000e+00,
           4.00000e+00, 8.00000e+00, 1.60000e+01, 3.20000e+01, 6.40000e+01,
           1.28000e+02, 2.56000e+02])},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print('Returned hyperparameter: {}'.format(clf3.best_params_))
print('Best classification accuracy in train is: {}'.format(clf3.best_score_))
print('Classification accuracy on test is: {}'.format(clf3.score(X_test, y_test)))
```

    Returned hyperparameter: {'svc__C': 0.015625}
    Best classification accuracy in train is: 0.863002432717575
    Classification accuracy on test is: 0.84
    

**Question:** What did we change with respect to the previous run of ```LinearSVC()```?

**Answer:**

A pipeline allows you to perform several operations in a row. First, we renormalize the features with `MaxAbsScaler` (using the training data), in order to put them on the same scale.

This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1. 

For each i-th component of each vector $(X_j)$, we probably divide by the highest value (in absolute value) that is to say $X'_{i,j}=\frac{X_{i,j}}{X_{i_{max},j}}$ with $i_{max}= \underset{j}{\max}|X_{i,j}|$ .

Second, we apply the same algorithm as before (svc, a LinearSVC) to fit the training data in a 3-fold CV validation (as before) to choose the best value of the C parameter which seems to be `C = 0.015625`. 

**Question:** Explain what happens if we execute
```python
    pipe.fit(X_train, y_train)
    pipe.predict(X_test, y_test)
```

**Answer:**
`pipe.fit` works. It fits the dataset as before but not using a cross-validation but using the default `C` parameter (that is to say ...) and `max_iter=5000`. 


`pipe.predict` returns the following error : 

`TypeError: predict() takes 2 positional arguments but 3 were given`

The function does not work here because when we do a prevision, we do not need to enter the `Y` values, we just need the `X` ones. 

This is why the following lines work (see below). 
```python
    pipe.predict(X_test) #working
    pipe.score(X_test, y_test)  #working
```


```python
pipe.fit(X_train, y_train)
```




    Pipeline(memory=None,
             steps=[('scaler', MaxAbsScaler(copy=True)),
                    ('svc',
                     LinearSVC(C=1.0, class_weight=None, dual=True,
                               fit_intercept=True, intercept_scaling=1,
                               loss='squared_hinge', max_iter=5000,
                               multi_class='ovr', penalty='l2', random_state=None,
                               tol=0.0001, verbose=0))],
             verbose=False)




```python
pipe.predict(X_test, y_test) #not working
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-20-ed47c41d3d19> in <module>
    ----> 1 pipe.predict(X_test, y_test) #not working
    

    ~\Anaconda3\lib\site-packages\sklearn\utils\metaestimators.py in <lambda>(*args, **kwargs)
        114 
        115         # lambda, but not partial, allows help() to work with update_wrapper
    --> 116         out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        117         # update the docstring of the returned function
        118         update_wrapper(out, self.fn)
    

    TypeError: predict() takes 2 positional arguments but 3 were given



```python
pipe.predict(X_test) #working
```




    array(['4', '1', '6', '5', '3', '4', '1', '3', '3', '1', '0', '6', '3',
           '4', '9', '7', '6', '4', '1', '6', '1', '4', '3', '8', '9', '4',
           '7', '8', '1', '1', '5', '6', '1', '4', '0', '2', '0', '9', '9',
           '6', '2', '4', '6', '4', '9', '8', '7', '7', '0', '9', '4', '6',
           '9', '7', '5', '2', '2', '7', '1', '6', '5', '4', '2', '8', '9',
           '6', '3', '2', '8', '1', '7', '0', '1', '3', '2', '0', '9', '0',
           '0', '0', '1', '0', '8', '7', '9', '9', '2', '1', '8', '9', '3',
           '1', '5', '1', '3', '1', '3', '0', '8', '7', '0', '6', '5', '9',
           '4', '0', '2', '5', '6', '9', '7', '5', '6', '3', '9', '7', '9',
           '0', '9', '3', '9', '1', '3', '1', '3', '6', '1', '3', '8', '8',
           '2', '9', '9', '6', '2', '7', '4', '3', '9', '2', '7', '0', '8',
           '1', '2', '3', '6', '0', '8', '1', '5', '0', '0', '3', '0', '4',
           '3', '1', '3', '9', '0', '4', '3', '9', '4', '8', '4', '7', '3',
           '0', '9', '5', '8', '4', '6', '6', '3', '0', '4', '7', '0', '3',
           '1', '8', '7', '8', '0', '4', '9', '6', '7', '1', '1', '2', '2',
           '3', '6', '6', '2', '0'], dtype=object)




```python
pipe.score(X_test, y_test)  #working
```




    0.835




```python
# Logistic regression
pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000))])
parameters4 = {'logreg__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
clf4 = GridSearchCV(pipe, parameters4, cv=3)
clf4.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('logreg',
                                            LogisticRegression(C=1.0,
                                                               class_weight=None,
                                                               dual=False,
                                                               fit_intercept=True,
                                                               intercept_scaling=1,
                                                               l1_ratio=None,
                                                               max_iter=5000,
                                                               multi_class='auto',
                                                               n_jobs=None,
                                                               penalty='l2',
                                                               random_state=None,
                                                               solver='lbfgs',
                                                               tol=...
                 iid='deprecated', n_jobs=None,
                 param_grid={'logreg__C': array([3.90625e-03, 7.81250e-03, 1.56250e-02, 3.12500e-02, 6.25000e-02,
           1.25000e-01, 2.50000e-01, 5.00000e-01, 1.00000e+00, 2.00000e+00,
           4.00000e+00, 8.00000e+00, 1.60000e+01, 3.20000e+01, 6.40000e+01,
           1.28000e+02, 2.56000e+02])},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print('Returned hyperparameter: {}'.format(clf4.best_params_))
print('Best classification accuracy in train is: {}'.format(clf4.best_score_))
print('Classification accuracy on test is: {}'.format(clf4.score(X_test, y_test)))
```

    Returned hyperparameter: {'logreg__C': 0.0078125}
    Best classification accuracy in train is: 0.8705039372205788
    Classification accuracy on test is: 0.84
    

**Question:** what is the difference between ```StandardScaler()``` and ```MaxAbsScaler()```? What are other scaling options available in ```sklearn```?

**Answer:**

- ```StandardScaler()```  : Standardize features by removing the mean and scaling to unit variance
The standard score of a sample x is calculated as:

$z = (x - u) / s$

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.
However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values. StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.

- ```MaxAbsScaler()``` : see previous question for the definition. 




The **differences** between these two methods are the following : 
* ```MaxAbsScaler()``` method does not shift/center the data, and thus does not destroy any sparsity, and thus can be applied to sparse CSR or CSC matrices, unlike ```StandardScaler()``` 
* ```MaxAbsScaler()``` rescales the data et such that the absolute values are mapped in the range $[0, 1]$, unlike ```StandardScaler()```
* On positive only data, ```MaxAbsScaler()``` behaves similarly to ``MinMaxScaler``` and therefore also suffers from the presence of large outliers. 


Other scaling options available in ```sklearn```:
1. ```MinMaxScaler()``` : rescales the data set such that all feature values are in the range $[0, 1]$ As StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.
```python
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```
2. ```RobustScaler()``` : Unlike the previous scalers, the centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers (robust to outliers).
3. ```Normalizer()``` :  The norm of each feature must be equal to 1. We can use many norms : $L^1$, $L^2$, $L^\infty$ ...

The whole list of preprocessing methods is available [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)


**Question:** using the previous code as an example achieve test accuracy $\geq 0.9$. You can use any method from sklearn package. Give a mathematical description of the selected method. Explain the range of considered hyperparamers.


**Answer:**

We give here the examples of two methods but there are plenty of them. 

1. Example 1 : SVC Classifier (other SVM classifier but not linear)

> The range of considered parameter $C$ is the same as above. 

Given training vectors $x_i \in \mathbb{R}^p$, i=1,…, n, in two classes, and a vector $y \in \{1, -1\}^n$ 
, our goal is to find $w \in \mathbb{R}^p$ and $b \in \mathbb{R}$ such that the prediction given by $\text{sign} (w^T\phi(x) + b)$ is correct for most samples.
SVC solves the following problem:
 \begin{align}\begin{aligned}\min_ {w, b, \zeta} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i\\\begin{split}\textrm {subject to } & y_i (w^T \phi (x_i) + b) \geq 1 - \zeta_i,\\
& \zeta_i \geq 0, i=1, ..., n\end{split}\end{aligned}\end{align} 

Intuitively, we’re trying to maximize the margin (by minimizing $||w||^2 = w^Tw$), while incurring a penalty when a sample is misclassified or within the margin boundary. Ideally, $the value y_i
(w^T \phi (x_i) + b)$ would be $\geq 1$ for all samples, which indicates a perfect prediction. But problems are usually not always perfectly separable with a hyperplane, so we allow some samples to be at a distance $\zeta_i$ from their correct margin boundary. The penalty term $C$ controls the strengh of this penalty (as seen above). 

The dual problem to the primal is

 \begin{align}\begin{aligned}\min_{\alpha} \frac{1}{2} \alpha^T Q \alpha - e^T \alpha\\\begin{split}
\textrm {subject to } & y^T \alpha = 0\\
& 0 \leq \alpha_i \leq C, i=1, ..., n\end{split}\end{aligned}\end{align} 

where $e$ is the vector of all ones, and $Q$ is an $n$ by $n$ positive semidefinite matrix, $Q_{ij} \equiv y_i y_j K(x_i, x_j)$, where $K(x_i, x_j) = \phi (x_i)^T \phi (x_j)$ is the kernel. The terms $\alpha_i$ are called the dual coefficients, and they are upper-bounded by $C$. This dual representation highlights the fact that training vectors are implicitly mapped into a higher (maybe infinite) dimensional space by the function $\phi$ (kernel trick).
    
2. Example 2 : Random forest

The RandomForest algorithm  is a perturb-and-combine technique specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.

As other classifiers, forest classifiers have to be fitted with two arrays: a sparse or dense array X of size [n_samples, n_features] holding the training samples, and an array Y of size [n_samples] holding the target values (class labels) for the training samples. Like decision trees, forests of trees also extend to multi-output problems (if Y is an array of size [n_samples, n_outputs]).

In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size `max_features`. The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.

In contrast to the original publication, the scikit-learn implementation combines classifiers by averaging their probabilistic prediction, instead of letting each classifier vote for a single class.

>   We can vary the range of several parameters. Here we chose to make XXXXX. Below the different parameters possibile : 
    * n_estimators = number of trees in the foreset
    * max_features = max number of features considered for splitting a node
    * max_depth = max number of levels in each decision tree
    * min_samples_split = min number of data points placed in a node before the node is split
    * min_samples_leaf = min number of data points allowed in a leaf node
    * bootstrap = method for sampling data points (with or without replacement)




```python
# Example 1 : SVC Classifier (other SVM classifier but not linear)
from sklearn.svm import SVC
svc2 = SVC(max_iter=5000) # by default : rbf kernel

pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', svc2)])
parameters5 = {'svc__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
clf5 = GridSearchCV(pipe, parameters5, cv=3)
clf5.fit(X_train, y_train)

print('Returned hyperparameter: {}'.format(clf5.best_params_))
print('Best classification accuracy in train is: {}'.format(clf5.best_score_))
print('Classification accuracy on test is: {}'.format(clf5.score(X_test, y_test)))
```

    Returned hyperparameter: {'svc__C': 8.0}
    Best classification accuracy in train is: 0.9190022106064085
    Classification accuracy on test is: 0.945
    


```python
#Example 2 : Random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV # Number of trees in random forest

from pprint import pprint# Look at parameters used by our current forest
rf = RandomForestRegressor(random_state = 42)
print('Parameters currently in use:\n')
pprint(rf.get_params())
```

    Parameters currently in use:
    
    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'criterion': 'mse',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': 42,
     'verbose': 0,
     'warm_start': False}
    


```python
rf = RandomForestClassifier(max_depth=10, random_state=0) # defining classifier
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
parameters = {'n_estimators': n_estimators}
clf6 = GridSearchCV(rf, parameters, cv=3) #cross-validation : method 3-fold.
clf6.fit(X_train, y_train)

print('Returned hyperparameter: {}'.format(clf6.best_params_))
print('Best classification accuracy in train is: {}'.format(clf6.best_score_))
print('Classification accuracy on test is: {}'.format(clf6.score(X_test, y_test)))
```

    Returned hyperparameter: {'n_estimators': 1200}
    Best classification accuracy in train is: 0.9115044579812196
    Classification accuracy on test is: 0.93
    

## Visualizing errors

Some ```sklearn``` methods are able to output probabilities ```predict_proba(X_test)```.

**Question** There is a mistake in the following chunk of code. Fix it.

**Answer:**

The line with a mistake is the following : 
```python
axes[1, j].bar(np.arange(10), clf4.predict_proba(image.reshape(1, -1)))  # MISTAKE !
``` 
If we execute the code line by line we find the following error 
``` only size-1 arrays can be converted to Python scalars ```

It means that the two following objects do not have the same dimensions : 

```python
print(np.arange(10))
print(clf4.predict_proba(image.reshape(1, -1)))
```
```
[0 1 2 3 4 5 6 7 8 9] # 1 dimension
[[1.19370882e-01 1.08367644e-04 7.49096695e-02 7.55611181e-01
  2.56621514e-06 4.47842619e-02 2.03011570e-04 2.06422609e-03
  2.94488853e-03 9.45386443e-07]] # 2 dimensions => must be 1
```

The line must be replaced by : 

```python
axes[1, j].bar(np.arange(10), clf4.predict_proba(image.reshape(1, -1))[0]) # CORRECTION ! 
```



```python
axes = plt.subplots(2, 4)[1] 

# More details about zip() function here https://docs.python.org/3.3/library/functions.html#zip
y_pred = clf4.predict(X_test)
j = 0 # Index which iterates over plots
for true_label, pred_label, image in list(zip(y_test, y_pred, X_test)):
    if j == 4: # We only want to look at 4 first mistakes
        break
    if true_label != pred_label:
        # Plotting predicted probabilities
        #axes[1, j].bar(np.arange(10), clf4.predict_proba(image.reshape(1, -1)))  # MISTAKE !
        axes[1, j].bar(np.arange(10), clf4.predict_proba(image.reshape(1, -1))[0]) # CORRECTION ! 
        axes[1, j].set_xticks(np.arange(10))
        axes[1, j].set_yticks([])
        
        # Plotting the image
        axes[0, j].imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        axes[0, j].set_title('Predicted {}'.format(pred_label))
        j += 1
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA2gAAAKOCAYAAADTfDoWAAAgAElEQVR4nOzdeZCdVZ3/8dNJOh2S0CSQEDoBAgghCK1hE2XJgksGYaIZGQYUBAEHUhBINAWiSBw2xwGDhY5YIgZwV8SlRMaNsJQOihQ1uCAOI7uKBCIQkkCW7+8Pft1J8yzd98nznPP9fu/7VXWqrE5uP09uf873nk/bfQkCAAAAAFAhpL4BAAAAAMArKGgAAAAAoAQFDQAAAACUoKABAAAAgBIUNAAAAABQgoIGAAAAAEpQ0AAAAABACQoaAAAAAChBQQMAAAAAJShoAAAAAKAEBQ0AAAAAlKCgAQAAAIASFDQAAAAAUIKCBgAAAABKUNAAAAAAQAkKGgAAAAAoQUEDAAAAACUoaAAAAACgBAUNAAAAAJSgoAEAAACAEhQ0AAAAAFCCggYAAAAASlDQAAAAAEAJChoAAAAAKEFBAwAAAAAlKGgAAAAAoAQFDQAAAACUoKABAAAAgBIUNAAAAABQgoIGAAAAAEpQ0AAAAABACQoaAAAAAChBQQMAAAAAJShoAAAAAKAEBQ0AAAAAlKCgAQAAAIASFDQAAAAAUIKCBgAAAABKUNAAAAAAQAkKGgAAAAAoQUEDAAAAACUoaAAAAACgBAUNAAAAAJSgoAEAAACAEhQ0AAAAAFCCggYAAAAASlDQAAAAAEAJChoAAAAAKEFBAwAAAAAlKGgAAAAAoAQFDQAAAACUoKABAAAAgBIUNAAAAABQgoIGAAAAAEpQ0AAAAABACQoaAAAAAChBQQMAAAAAJShoAAAAAKAEBQ0AAAAAlKCgAQAAAIASFDQAAAAAUIKCBgAAAABKUNAAAAAAQAkKGgAAAAAoQUEDAAAAACUoaAAAAACgBAUNAAAAAJSgoAEAAACAEhQ0AAAAAFCCggYAAAAASlDQAAAAAEAJChoAAAAAKEFBAwAAAAAlKGgAAAAAoAQFDQAAAACUoKABAAAAgBIUNAAAAABQgoIGAAAAAEpQ0AAAAABACQoaAAAAAChBQQMAAAAAJShoAAAAAKAEBQ0AAAAAlKCgAQAAAIASFDQAAAAAUIKCBgAAAABKUNAAAAAAQAkKGgAAAAAoQUEDAAAAACUoaFBn6dKlEkKQpUuXDvj4ihUrJIQgs2bNSnJfdVm+fLmEEOTkk09OfStoAPmFdWQYlpFfeEBBc27WrFkSQhiwhg8fLhMmTJA3v/nNsnz5ctm4cWPq2xwgxnBdsWKFLF26VFasWLHVn6tVVYfrN7/5TXn/+98vBx54oPT09EhnZ6eMHTtWZsyYIR/+8Iflb3/7WzM3nBD5zWcxv33PS9m69dZbm7nphMhwPjJsA/nNZzG/7XiGsIyC5lzfcN1ll13ksMMOk8MOO0wOPPBA6e7u7h+2Rx11lLz88supb7Vf0XD95S9/KXvvvbecdNJJjV0jhqrD9fWvf72EEKSrq0t22203Oeigg2TXXXft/zpOmDBB7rvvvmZuOhHy29o1Ytjaw+2WX8tXr7vvvruZm06IDLd2jRjI8NCR39auEQNniPZAQXOub7i+eoisX79eLr/88v6NecUVV6S5wRwxBp/F4fr5z39e7rjjjswL4f333y/77befhBBkn332qfFO0yO/6a5RZGsPtynuOSUynO4aRcjw0JHfdNcowhmiPVDQnCsarn2OPvpoCSHI61//+rg3VoLh2rpf/vKX/S+Uv//972v7vKmR33TXKMLhtjVkON01ipDhoSO/6a5RhDNEe6CgOTfYcL3yyislhCDbbLNN/8e2/DntDRs2yFVXXSUzZsyQsWPHSgjZyHz/+9+Xo48+WiZOnCidnZ0yefJkOfHEE+V3v/td4X0999xzsmTJEpk6dap0dXXJrrvuKgsXLpRnnnmmcPAN9vPja9eulauvvloOP/xwGT9+fP//jf/Od75TvvGNb/T/vbLfH8j73Hfeeacce+yx/T+zveOOO8r8+fPlF7/4ReG/b926dXLJJZfItGnTpKurS3p6euSUU06RRx99tJHh+vzzz/f/G37961/X9nlTI79+8tuOh1sRMkyGbSO/fvJbxusZwjIKmnODDdf/+I//kBCCjB49uv9jfUNs5syZMm/ePAkhyG677SYHHnigbLvttv1/b8OGDXLKKaf0b+qJEyfK/vvvL9ttt52EEGTUqFHywx/+MHPNlStXyr777ishBOno6JD99ttP9ttvP+no6JDXvOY1cs4557Q8XJ988kl53ete138vfT9fPWnSJAkhyHbbbdf/dw877DDZZZddJITs7xKcffbZAz7vhRde2P85x48fL/vvv79MmDBBQggybNgw+cIXvpC5l7Vr1w74xepp06bJjBkzZMSIEbLDDjvIRz/60dqH66233iohBBk7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FSvDVfIr4jPDzOB8mmcw+R2IGZzFDEZsFDRnbrnlFgkhyOmnn575s77vVP3whz8c9PMMdbgW2bhxo4wePVrGjBnT8mPnz58vIQR56KGHSv/escceK1OnTpXVq1fXPlxXr14t3d3dMnv27CE/psx73/teCSHIN77xjdw/v/baayWEIJ/+9Kczf3b55ZdLCEG+9a1vDfj4e97znsKv5wc+8AEJIchPfvKTWu4/pjoy3O75FYmb4Sr5FfGZYWZwPs0zmPwOxAzOYgYjNgqaM33DM28D//SnP5UQglxwwQWDfp46hmt3d7dst912LT3umWeekSlTpkh3d3fud8f63HTTTRJCkP/6r/8SERnScH3d614nn/vc5+Syyy6T6667Th555JHCv3/zzTf3D7vnnntObrzxRrn88svl+uuvl6eeeqqlf9MLL7wgY8aMkfHjx8u6dety/84vfvGL/u9+banvu19dXV3y+OOPD/izt7zlLRJCkN///veZz3f11VdLCEE+8pGPtHSvGtSR4XbPr0jcDFfJr4jPDDOD82meweR3IGZwFjMYsVHQnHnXu94lIYTcX+585JFHJIQg//zP/zzo59na4do3mI499tjSv7dq1SpZunSpfPSjH5XTTz9ddtxxRxkxYkThG2mIiKxcuVImTZokJ554Yv/HhjJcX72GDx8uixYtyv257I985CMSQpCLL75YdtpppwGPGzNmjHz5y18ewrPwiuuuu05CCHLWWWeV/r2+747NmDFDFi9eLIsWLZJ99tlHtt9++9zvfB1//PESQsj9fYC+73wdd9xxQ75PLerIcLvnVyR+hlvNr4jPDDOD82mfweR3M2ZwFjMYsVHQnHnrW98qIQT53//938yfPfvssxJCkLe97W2Dfp6tGa5//etfZcqUKdLV1SW/+93vSv/uww8/PGBojR07dtCh9e53v1smTpwoTz/9dP/HBhuuS5Yskf/+7/+WZ599Vp555hm55ZZbZN9995UQgnz0ox/N/P0zzjijfwAfc8wx8oc//EH+/ve/y9e//nUZN26cjBgxQu67775BnolXHHHEERJCkF//+telf2/Tpk1y4YUXSkdHR//z0dHRIQsWLBjwb+1z/fXX9//YxZbfUXvsscf6f1F6KF9rberIcLvnVyR+hlvNr4jPDDOD82mfweR3M2ZwFjMYsVHQnOn7v6vzfvY6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" width="639.4666666666667">


## Changing the Loss function

It often happens that the accuracy is not the right way to evaluate the performance. ```sklearn``` has a large variety of other metrics both in classification and regression. See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

Here we want to understand how to change the cross-validation metric with minimal effort.


```python
# SVM Classifier + Pipeline + New score function

pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', svc)])
parameters4 = {'svc__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
balanced_scorer = make_scorer(balanced_accuracy_score)

clf4 = GridSearchCV(pipe, parameters3, cv=3, scoring=balanced_scorer)
clf4.fit(X_train, y_train)

print('Returned hyperparameter: {}'.format(clf4.best_params_))
print('Best Balanced accuracy in train is: {}'.format(clf4.best_score_))
print('Balanced accuracy on test is: {}'.format(clf4.score(X_test, y_test)))
```

    Returned hyperparameter: {'svc__C': 0.015625}
    Best Balanced accuracy in train is: 0.8612334093654231
    Balanced accuracy on test is: 0.825627008328415
    

**Question:** What is ```balanced_accuracy_score```? Write its mathematical mathematical description.


**Answer:** 

The ```balanced_accuracy_score``` function computes the balanced accuracy, which avoids inflated performance estimates on imbalanced datasets. It is the macro-average of recall scores per class or, equivalently, raw accuracy where each sample is weighted according to the inverse prevalence of its true class. Thus for balanced datasets, the score is equal to accuracy.

In the *binary case*, balanced accuracy is equal to the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate), or the area under the ROC curve with binary predictions rather than scores:

$$
\texttt{balanced-accuracy} = \frac{1}{2}\left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right )
$$

If the classifier performs equally well on either class, this term reduces to the conventional accuracy (i.e., the number of correct predictions divided by the total number of predictions).

In contrast, if the conventional accuracy is above chance only because the classifier takes advantage of an imbalanced test set, then the balanced accuracy, as appropriate, will drop to $\frac{1}{n_{classes}}$

The score ranges from 0 to 1, or when `adjusted=True` is used, it rescaled to the range $\frac{1}{1-n_{classes}}$
to 1, inclusive, with performance at random scoring 0.

If $y_i$ is the true value of the $i$-th sample, and $w_i$ is the corresponding sample weight (number of observations of the $i$-th class out of the total number of observations), then we adjust the sample weight to :

$$
\hat{w}_i = \frac{w_i}{\sum_j{1(y_j = y_i) w_j}}
$$

where $1(x)$ is the indicator function. Given predicted $\hat{y}_i$ for sample $i$, balanced accuracy is defined as:

$$
balanced-accuracy(y,\hat y,w)=\frac{1}{\sum \hat w_i} \sum_i 1_{(\hat y_i = y_i)}\hat w_i
$$

With `adjusted=True`, balanced accuracy reports the relative increase from $\texttt{balanced-accuracy}(y, \mathbf{0}, w) =
\frac{1}{n\_classes}$. In the binary case, this is also known as *Youden’s J statistic*, or informedness.

*Source: [here](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score).*


**In a nutshell, this function allows to take into account the case when somme classes are over or under-represented (= unbalanced data).**




```python
#Let's go back to our simple example for a multiclass analysis to illustrate
#and calculate the balanced-accuracy
y_true = [0, 0, 1, 2, 3]
y_pred = [0, 1, 2, 1, 3]
print("balanced-accuracy : ", balanced_accuracy_score(y_true, y_pred))
```

    balanced-accuracy :  0.375
    

Indeed, 

$$
balanced-accuracy(y,\hat y,w)=\frac{1}{\sum \hat w_i} \sum_i 1_{(\hat y_i = y_i)}\hat w_i = \frac{1}{0.5 + 0.5 + 1 + 1 + 1} \sum_i 1_{(\hat y_i = y_i)}\hat w_i = \frac{1}{4} (0.5 + 1) = \frac{3}{8} = 0.375
$$


Sometimes it is important to look at the confusion matrix of the prediction.

**Question:** What is the confusion matrix? What are the conclusions that we can draw from the ```confusion_matrix(y_test, clf4.predict(X_test))```

**Answer:** 

By definition a confusion matrix $C$ is such that  $C_{i,j}$ is equal to the number of observations known to be in group $i$ and predicted to be in group $j$.

Here, when can see for example that : 

* 0 are well identified (all predicted as being 0's)
* 3 are sometimes (3 times out of 23) identified as 5
* 8 are also sometimes (3 times out of 17) identified as 5


```python
print(confusion_matrix(y_test, clf4.predict(X_test)))
```

    [[22  0  0  0  0  0  0  0  0  0]
     [ 0 24  0  0  0  0  0  0  2  0]
     [ 0  0 14  1  1  0  0  0  0  0]
     [ 0  0  0 18  0  3  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 1  0  0  1  0  6  0  1  0  1]
     [ 1  2  1  0  0  0 20  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  2  0  1  0  3  0  0 11  0]
     [ 0  0  0  0  2  0  0  2  1 21]]
    

# PART 2 -- Problem

The data that we have contains images with $10$ classes. Normally, accuracy is a reasonable choice of the loss function to be optimized, but in this problem we *really* do not like when digits from $\{5, 6, 7, 8, 9\}$ are predicted to be from $\{0, 1, 2, 3, 4\}$.

When writing your report on this part, include:
   1. description of your loss function
   2. description of the pipeline
   3. description of the algorithms that you used 

## First thought

**Question:** Propose a loss function that would address our needs. Explain your choice.

**Answer:**

For this problem, our **FIRST IDEA** was to define two sets :

1. Class 1 = {0,1,2,3,4} 
2. Class 0 = {5,6,7,8,9} 

In order to find a proper loss function, we use the definition of the precision score.
In fact, we want to minimize Y_pred in class  when Y_true in class 0 (reduce False Positive rate).

In a binary classification task, the terms ‘’positive’’ and ‘’negative’’ refer to the classifier’s prediction, and the terms ‘’true’’ and ‘’false’’ refer to whether that prediction corresponds to the external judgment (sometimes known as the ‘’observation’’). Given these definitions, we can formulate the following table:


|  | Actual : Class 1                                                     | Actual : Class 0                                                     |   |   |
|--------------------------|-------------------------------------------------------------|-------------------------------------------------------------|---|---|
| Predicted : Class 1                  | TP (True Positive) Y_true and Y_pred in Class 1             | FP (False Positive) Y_true in Class 0 and Y_pred in Class 1 |   |   |
| Predicted : Class 0                  | FN (False Negative) Y_true in Class 1 and Y_pred in Class 0 | TN (True Negative) Y_true and Y_pred in Class 0             |   |   |



In this context, we can define the notions of precision as : $\text{precision} = \frac{tp}{tp + fp},$ which is the indicator which is the most interesting to answer to this problem.

 **=> After training the model, we compare our results with the previous confusion matrix to verify that the bottom right square has evolved well : the sum of the last five lines and the first five columns should be smaller.**


```python
#Define New precision score function with 2 class : Class 1 = {0,1,2,3,4} and Class 0 = {5,6,7,8,9}
def custom_precision_score(y_true, y_pred):
    #Class 1 = {0,1,2,3,4} and Class 0 = {5,6,7,8,9}
    #Calcul TP (True Positive) = y_true and y_pred in class 1
    true_positive = np.sum((y_true.astype(int) < 5 ) & (y_pred.astype(int) < 5))
    #Calcul FP (False Positive) = y_true in class 0 and y_pred in class 1
    false_positive = np.sum((y_true.astype(int) > 4) & (y_pred.astype(int) < 5))
    return true_positive / (true_positive + false_positive)
```

**Question:** Following above examples, make an ML pipeline that uses *your* loss function and finds appropriate classifiers.




```python
def use_svc(model):
    #define the scorer
    scorer = make_scorer(custom_precision_score, greater_is_better=True)
    pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', model)])
    parameters = {'svc__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
    clf = GridSearchCV(pipe, parameters, cv=3, scoring=scorer)
    clf.fit(X_train, y_train)
    return(clf)
```


```python
#Tests on different models
def evaluation_model(model):
    grid=use_svc(model)
    print('{} -- Returned hyperparameter: {}'.format(model, grid.best_params_))
    print('{} -- Best accuracy in train is: {}'.format(model, grid.best_score_))
    print('{} -- Accuracy on test is: {}'.format(model, grid.score(X_test, y_test)))
    best_model = grid.best_estimator_
    print('{} -- Best Estimator in train is: {}'.format(model, grid.best_estimator_))
    y_pred = best_model.predict(X_test)
    print('{} -- Custom precision score in test is: {}'.format(model, custom_precision_score(y_test, y_pred)))
    print('{} -- Confusion matrix: \n {}'.format(model, confusion_matrix(y_test, y_pred)))
```


```python
svc_linear = LinearSVC(max_iter=5000)
svc = SVC(max_iter=5000)

dict_of_models = {'Linear SVC': svc_linear,
                  'SVC': svc
                  }
for name, model in dict_of_models.items():
    evaluation_model(model)
```

    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Returned hyperparameter: {'svc__C': 0.125}
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Best accuracy in train is: 0.9222826688026527
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Accuracy on test is: 0.9279279279279279
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Best Estimator in train is: Pipeline(memory=None,
             steps=[('scaler', MaxAbsScaler(copy=True)),
                    ('svc',
                     LinearSVC(C=0.125, class_weight=None, dual=True,
                               fit_intercept=True, intercept_scaling=1,
                               loss='squared_hinge', max_iter=5000,
                               multi_class='ovr', penalty='l2', random_state=None,
                               tol=0.0001, verbose=0))],
             verbose=False)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Custom precision score in test is: 0.9279279279279279
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=5000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0) -- Confusion matrix: 
     [[22  0  0  0  0  0  0  0  0  0]
     [ 0 23  0  2  0  0  0  0  0  1]
     [ 1  0 14  1  0  0  0  0  0  0]
     [ 0  0  1 20  0  0  0  0  1  1]
     [ 0  1  1  0 17  0  0  0  1  0]
     [ 1  0  0  1  0  7  0  1  0  0]
     [ 1  0  0  0  0  1 20  0  1  1]
     [ 0  0  0  0  1  0  0 14  0  1]
     [ 0  1  0  1  0  2  1  0 12  0]
     [ 0  0  0  0  2  0  0  2  1 21]]
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Returned hyperparameter: {'svc__C': 2.0}
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Best accuracy in train is: 0.9379540095272763
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Accuracy on test is: 0.9629629629629629
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Best Estimator in train is: Pipeline(memory=None,
             steps=[('scaler', MaxAbsScaler(copy=True)),
                    ('svc',
                     SVC(C=2.0, break_ties=False, cache_size=200, class_weight=None,
                         coef0=0.0, decision_function_shape='ovr', degree=3,
                         gamma='scale', kernel='rbf', max_iter=5000,
                         probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False))],
             verbose=False)
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Custom precision score in test is: 0.9629629629629629
    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=5000, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) -- Confusion matrix: 
     [[22  0  0  0  0  0  0  0  0  0]
     [ 0 24  1  1  0  0  0  0  0  0]
     [ 0  0 16  0  0  0  0  0  0  0]
     [ 0  0  0 21  0  1  0  0  0  1]
     [ 0  0  1  0 18  0  0  0  0  1]
     [ 0  0  0  0  0 10  0  0  0  0]
     [ 0  1  1  0  0  0 22  0  0  0]
     [ 0  0  0  0  0  0  0 16  0  0]
     [ 0  1  0  1  0  0  0  0 15  0]
     [ 0  0  0  0  0  0  0  1  1 24]]
    

<mark>
Isabelle : Peaufiner la mise en forme des résultats et conclure. Par ailleurs, je te poserai des questions à l'oral durant le point ! 
</mark>
    

**=> However, this method does not allow to penalize the error of classification inside the two classes (1 and 0). For example, if I predict a 0 instead of a 1 it is not penalized in the loss function whereas it should be.**



##  Second thought

**Question:** Propose a loss function that would address our needs. Explain your choice.

**Answer:**

Our **SECOND IDEA** was to change the loss function in another way to take into account the fact that we must penalize if we predict the wrong number even if the actual or predicted values are both small or large


In the usual classification that we made above, we used the following loss 
$$l_1(y,\hat{y}) = 1_{\hat{y} \ne y}$$

Here, we can modify a little bit to show that we do not like if $y \in H = \{5, 6, 7, 8, 9\}$ and $\hat{y} \in L = \{0, 1, 2, 3, 4\}$. 

$$
l_2(y,\hat{y}) = 1_{(\hat{y} \ne y) \& [(y \notin H) \text{ or } (\hat{y} \notin L)]} + \alpha * 1_{(\hat{y} \ne y) \& (y \in H) \& (\hat{y} \in L)}
$$

where $\alpha > 1$ reflects the aversion that you have when $y \in H$ and $\hat{y} \in L$.



```python
import pandas as pd

# this function returns a vector containing the loss for each pair of (y_true, y_pred)
def my_custom_loss_func(y_true,y_pred, alpha):
    df = pd.DataFrame({'y_true': [int(s) for s in y_true],'y_pred': [int(s) for s in y_pred]})
    df['loss'] = 1* (df['y_true'] != df['y_pred'])
    df['loss'] = df['loss'] + (alpha-1)*((df['y_true'] >4) & (df['y_pred'] < 5))
    return df['loss']
```


```python
# an example to illustrate it
y_true = ['9', '9', '9'] #we have 3 nines in the dataset
y_pred = ['9', '8', '1'] 
#the first one is well-predicted (loss 0),
#the second one is bad predicted but still high (loss 1)
#the last one is very bad predicted (with a low number) (loss alpha=2)
print(my_custom_loss_func(y_true,y_pred, alpha=2))
```

    0    0
    1    1
    2    2
    Name: loss, dtype: int32
    

Then, we need to define the scoring parameter to evaluate the predictions on the test set. All scorer objects follow the convention that higher return values are better than lower return values.

Previously we saw, two kinds of scorers : 

* The Accuracy classification score.
* The Balanced accuracy classification score. 



Here, we try to adapte theses scorers to take into account that we really don't like when $y \in H$ and $\hat{y} \in L$. We propose two new scorers : 

* The Accuracy "2" classification score. 
$$
\texttt{accuracy}_{2}(y, \hat{y}) = 1- (\frac{1}{\alpha * n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} l_2(y,\hat{y})) = 1- (\frac{1}{\alpha * n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1_{(\hat{y} \ne y) \& [(y \notin H) \text{ or } (\hat{y} \notin L)]} + \alpha * 1_{(\hat{y} \ne y) \& (y \in H) \& (\hat{y} \in L)}) 
$$

* The Balanced accuracy "2" classification score. 

$$
balanced-accuracy_2(y,\hat y,w)=1 - (\frac{1}{\alpha * \sum \hat w_i} \sum_i l_2(y,\hat{y}) \hat w_i)
$$

with still
$$
\hat{w}_i = \frac{w_i}{\sum_j{1(y_j = y_i) w_j}}
$$






```python
#both functions evaluate the quality of the prediction.

#accuracy_2 function
def accuracy_2(y_true,y_pred, alpha):
    nsamples = len(y_true)
    loss = my_custom_loss_func(y_true,y_pred,alpha)
    score = 1-((1/(alpha*nsamples)) * sum(loss))
    return score

#balanced_accuracy_2 function
def balanced_accuracy_2(y_true,y_pred, alpha):
    C = confusion_matrix(y_true, y_pred, sample_weight=None)    
    with np.errstate(divide='ignore', invalid='ignore'):
        wi_hat_par_cat = 1/C.sum(axis=1)
    if np.any(np.isinf(wi_hat_par_cat)):
        wi_hat_par_cat = wi_hat_par_cat[~np.isinf(wi_hat_par_cat)]    

    wi_hat= [None] * len(y_true)
    for i in range(len(y_true)):     #np.unique(y_true):
        for j in range(len(wi_hat_par_cat)): #np.unique(y_true): 
            if y_true[i]==np.unique(y_true)[j]: 
                wi_hat[i]=wi_hat_par_cat[j]
    loss = my_custom_loss_func(y_true,y_pred,alpha)
    score = 1-((1/(alpha*sum(wi_hat))) * sum(loss*wi_hat))
    return(score)

```


```python
# going back to the previous example to illustrate it
y_true = ['9', '9', '9'] 
y_pred = ['9', '8', '1'] 
print("accuracy_2 : ", accuracy_2(y_true,y_pred,alpha=2))
print("balanced_accuracy_2 : ", balanced_accuracy_2(y_true,y_pred,alpha=2)) 
```

    accuracy_2 :  0.5
    balanced_accuracy_2 :  0.5
    

**Question:** Following above examples, make an ML pipeline that uses *your* loss function and finds appropriate classifiers.

**Answer:** 

Now that we have defined the loss and the score functions, let's try to use this score with the 3 different methods (KNN, LinearSVC and LogisticRegression) of machine learning we discovered for this TP and evaluate them. Note that we still want that the sum of the last five lines and the first five columns of the confusion matrix should be smaller than before.

**=> Whereas the change of the loss doesn't change the model obtained with the knn methods, with the two other ones, we obtained different results from the previous models, with a confusion matrix evolving in the good way (smaller sum of the bottom left quarter). See details in the code.**


```python
# function which calculates the sum of the last five lines and five first columns of C
def sum_unwanted(clf):
    C=confusion_matrix(y_test, clf.predict(X_test))
    res = sum(C[5][:4])+sum(C[6][:5])+sum(C[7][:6])+sum(C[8][:7])+sum(C[9][:8])
    return(res)
```


```python
# function which evaluates the model
def evaluation_model(clf):
    print('Returned hyperparameter: {}'.format(clf.best_params_))
    print('Best classification accuracy2 in train is: {}'.format(clf.best_score_))
    print('Classification accuracy2 on test is: {}'.format(clf.score(X_test, y_test)))
    print('Confusion matrix: \n', confusion_matrix(y_test, clf.predict(X_test)))
```

### KNN


```python
def use_knn(alpha, balanced=False):
    if not balanced:
        scorer = make_scorer(accuracy_2, alpha=alpha)
    else:
        scorer = make_scorer(balanced_accuracy_2, alpha=alpha)   
    knn = KNeighborsClassifier() # defining classifier
    parameters = {'n_neighbors': [1, 2, 3, 4, 5]} # defining parameter space
    clf = GridSearchCV(knn, parameters, cv=3, scoring=scorer)
    clf.fit(X_train, y_train)
    return(clf)
```

Si $\alpha = 1$, we logically have the same results as the beginning of this TP: 


```python
clf7_alpha1 = use_knn(alpha=1, balanced=False)
evaluation_model(clf7_alpha1)
```

    Returned hyperparameter: {'n_neighbors': 1}
    Best classification accuracy2 in train is: 0.891497944721333
    Classification accuracy2 on test is: 0.875
    Confusion matrix: 
     [[21  0  0  0  0  0  1  0  0  0]
     [ 0 26  0  0  0  0  0  0  0  0]
     [ 0  0 14  0  0  2  0  0  0  0]
     [ 0  0  0 19  0  2  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 0  0  0  0  1  7  1  0  1  0]
     [ 0  0  0  0  0  1 23  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  1  0  1  0  0  0  0 14  1]
     [ 1  1  0  0  2  0  0  3  0 19]]
    

We try to increase the value of $\alpha$ that is to say to penalize more when $y \in H$ and $\hat{y} \in L$. let's try $\alpha=10$. 


```python
clf7_alpha10 = use_knn(alpha=10, balanced=False)
evaluation_model(clf7_alpha10)
```

    Returned hyperparameter: {'n_neighbors': 1}
    Best classification accuracy2 in train is: 0.9590000045022534
    Classification accuracy2 on test is: 0.9515
    Confusion matrix: 
     [[21  0  0  0  0  0  1  0  0  0]
     [ 0 26  0  0  0  0  0  0  0  0]
     [ 0  0 14  0  0  2  0  0  0  0]
     [ 0  0  0 19  0  2  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 0  0  0  0  1  7  1  0  1  0]
     [ 0  0  0  0  0  1 23  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  1  0  1  0  0  0  0 14  1]
     [ 1  1  0  0  2  0  0  3  0 19]]
    

The accuracy is better than the previous one but, be careful, we cannot compare the two accuracies because they are not using the same formula because it depends on the value of $\alpha$.

Unfortunately, the confusion matrix does not change because the best model remains the same (`n_neighbours = 1`) and do not decrease. However, this is totally normal because the parameter `n_neighbours = 1` already corresponds to the minimum :


```python
for n in [1, 2, 3, 4, 5]:
    knn = KNeighborsClassifier(n_neighbors = n);
    knn.fit(X_train, y_train)
    print('With parameter ',n, 'the sum of the left-bottom quarter of C is ', sum_unwanted(knn))
```

    With parameter  1 the sum of the left-bottom quarter of C is  10
    With parameter  2 the sum of the left-bottom quarter of C is  17
    With parameter  3 the sum of the left-bottom quarter of C is  15
    With parameter  4 the sum of the left-bottom quarter of C is  14
    With parameter  5 the sum of the left-bottom quarter of C is  13
    

### LinearSVC


```python
def use_svc(alpha, balanced=False,linear=True):
    #define the model
    if linear:
         model = LinearSVC(max_iter=5000)
    else:
         model =  SVC(max_iter=5000)
    #define the scorer
    if not balanced:
        scorer = make_scorer(accuracy_2, alpha=alpha)
    else:
        scorer = make_scorer(balanced_accuracy_2, alpha=alpha)  
        
    pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', model)])
    parameters = {'svc__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
    clf = GridSearchCV(pipe, parameters, cv=3, scoring=scorer)
    clf.fit(X_train, y_train)
    return(clf)
```

Example of the LinearSVC model with a balanced_accuracy function. With $\alpha = 1$ the results are obviously the same as previously that is to say : 


```python
clf8_alpha1 = use_svc(alpha=1, balanced=True,linear=True)
evaluation_model(clf8_alpha1)
```

    Returned hyperparameter: {'svc__C': 0.015625}
    Best classification accuracy2 in train is: 0.8612334093654243
    Classification accuracy2 on test is: 0.8256270083284148
    Confusion matrix: 
     [[22  0  0  0  0  0  0  0  0  0]
     [ 0 24  0  0  0  0  0  0  2  0]
     [ 0  0 14  1  1  0  0  0  0  0]
     [ 0  0  0 18  0  3  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 1  0  0  1  0  6  0  1  0  1]
     [ 1  2  1  0  0  0 20  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  2  0  1  0  3  0  0 11  0]
     [ 0  0  0  0  2  0  0  2  1 21]]
    

We can notice that, here, the parameter of the initial model `C=0.015625`is not the one which minimizes the sum of the bottom left quarter of the confusion matrix. The minimum is $13$ for `C=0.125`.


```python
for c in np.logspace(-8, 8, 17, base=2):
    model = LinearSVC(max_iter=5000,C=c)
    pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', model)])
    pipe.fit(X_train, y_train)
    print('With parameter ',c, 'the sum of the left-bottom quarter of C is ', sum_unwanted(pipe))
```

    With parameter  0.00390625 the sum of the left-bottom quarter of C is  18
    With parameter  0.0078125 the sum of the left-bottom quarter of C is  17
    With parameter  0.015625 the sum of the left-bottom quarter of C is  17
    With parameter  0.03125 the sum of the left-bottom quarter of C is  14
    With parameter  0.0625 the sum of the left-bottom quarter of C is  15
    With parameter  0.125 the sum of the left-bottom quarter of C is  13
    With parameter  0.25 the sum of the left-bottom quarter of C is  14
    With parameter  0.5 the sum of the left-bottom quarter of C is  15
    With parameter  1.0 the sum of the left-bottom quarter of C is  14
    With parameter  2.0 the sum of the left-bottom quarter of C is  16
    With parameter  4.0 the sum of the left-bottom quarter of C is  16
    With parameter  8.0 the sum of the left-bottom quarter of C is  16
    With parameter  16.0 the sum of the left-bottom quarter of C is  16
    

    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    

    With parameter  32.0 the sum of the left-bottom quarter of C is  16
    

    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    

    With parameter  64.0 the sum of the left-bottom quarter of C is  16
    With parameter  128.0 the sum of the left-bottom quarter of C is  16
    With parameter  256.0 the sum of the left-bottom quarter of C is  16
    

    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    

With $\alpha = 1000$, we get the expected result because the best model is now the one with `C=0.125` and the confusion matrix did changed in the good way.


```python
clf8_alpha1000 = use_svc(alpha=1000, balanced=True,linear=True)
evaluation_model(clf8_alpha10)
```

    Returned hyperparameter: {'svc__C': 0.015625}
    Best classification accuracy2 in train is: 0.9484628402030914
    Classification accuracy2 on test is: 0.9211322709685881
    Confusion matrix: 
     [[22  0  0  0  0  0  0  0  0  0]
     [ 0 24  0  0  0  0  0  0  2  0]
     [ 0  0 14  1  1  0  0  0  0  0]
     [ 0  0  0 18  0  3  0  0  1  1]
     [ 0  1  0  0 17  0  0  0  0  2]
     [ 1  0  0  1  0  6  0  1  0  1]
     [ 1  2  1  0  0  0 20  0  0  0]
     [ 0  0  0  0  1  0  0 15  0  0]
     [ 0  2  0  1  0  3  0  0 11  0]
     [ 0  0  0  0  2  0  0  2  1 21]]
    

### Logistic regression


```python
def use_logistic(alpha, balanced=False):
    #define the scorer
    if not balanced:
        scorer = make_scorer(accuracy_2, alpha=alpha)
    else:
        scorer = make_scorer(balanced_accuracy_2, alpha=alpha)  
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000))])
    parameters = {'logreg__C': np.logspace(-8, 8, 17, base=2)} # defining parameter space
    clf = GridSearchCV(pipe, parameters, cv=3, scoring=scorer)
    clf.fit(X_train, y_train)
    return(clf)
```

Example of the LogisticRegression model with a accuracy function. First, with $\alpha = 1$ :


```python
clf9_alpha1 = use_logistic(alpha=1, balanced=True)
evaluation_model(clf9_alpha1)
```

    Returned hyperparameter: {'logreg__C': 0.0078125}
    Best classification accuracy2 in train is: 0.8692423758419983
    Classification accuracy2 on test is: 0.8337791822414583
    Confusion matrix: 
     [[22  0  0  0  0  0  0  0  0  0]
     [ 0 21  0  3  0  0  0  0  2  0]
     [ 0  0 13  1  1  0  1  0  0  0]
     [ 0  0  1 17  0  3  0  0  1  1]
     [ 0  1  0  0 18  0  0  0  0  1]
     [ 1  0  0  0  0  8  0  1  0  0]
     [ 1  1  1  0  0  0 20  0  1  0]
     [ 0  0  0  0  1  0  0 14  0  1]
     [ 0  2  0  1  0  3  0  0 11  0]
     [ 0  0  0  0  0  0  0  2  0 24]]
    

We can notice that, here, the parameter of the initial model `0.0078125` already the one which minimizes the sum of the bottom left quarter of the confusion matrix. The minimum is also the same ($13$) for `C=0.00390625`  and `C=0.25`.


```python
for c in np.logspace(-8, 8, 17, base=2):
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000,C = c))])
    pipe.fit(X_train, y_train)
    print('With parameter ',c, 'the sum of the left-bottom quarter of C is ', sum_unwanted(pipe))
```

    With parameter  0.00390625 the sum of the left-bottom quarter of C is  13
    With parameter  0.0078125 the sum of the left-bottom quarter of C is  13
    With parameter  0.015625 the sum of the left-bottom quarter of C is  16
    With parameter  0.03125 the sum of the left-bottom quarter of C is  15
    With parameter  0.0625 the sum of the left-bottom quarter of C is  14
    With parameter  0.125 the sum of the left-bottom quarter of C is  14
    With parameter  0.25 the sum of the left-bottom quarter of C is  13
    With parameter  0.5 the sum of the left-bottom quarter of C is  14
    With parameter  1.0 the sum of the left-bottom quarter of C is  14
    With parameter  2.0 the sum of the left-bottom quarter of C is  14
    With parameter  4.0 the sum of the left-bottom quarter of C is  15
    With parameter  8.0 the sum of the left-bottom quarter of C is  15
    With parameter  16.0 the sum of the left-bottom quarter of C is  15
    With parameter  32.0 the sum of the left-bottom quarter of C is  15
    With parameter  64.0 the sum of the left-bottom quarter of C is  15
    With parameter  128.0 the sum of the left-bottom quarter of C is  15
    With parameter  256.0 the sum of the left-bottom quarter of C is  15
    


```python
## Convertir en Markdown pour le rapport latex : copier coller dans le terminal Jupyter. 
#cd "C:\Users\Kim Antunez\Desktop\3A\ML\DM_ML\TP1"
#jupyter nbconvert --to markdown TP1_KA_IB.ipynb
```
