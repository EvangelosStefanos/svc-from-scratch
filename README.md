# svc-from-scratch
  Support Vector Machine Classifier from scratch

## Details
  An implementation of a support vector machine classifier written in python for multiclass classification problems. Internally the [cvxopt](https://cvxopt.org/) package is used to solve the quadratic programming optimization problem. Support for the multiclass case is provided by the one-vs-all strategy.

## Evaluation
  The implementation is tested against the [SVC class from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). There are two test cases implemented, one for the [mnist dataset]() and one for the [gtzan dataset](). Each test case contains four trials, each with different parameter combinations. The results can be found in [mnist.csv](logs/mnist.csv) and [gtzan.csv](logs/gtzan.csv).

## Observations
  - The algorithm is significantly slower and does not scale well at all. Unsurprising, considering the fact that a N x N (where N is the number of training samples) kernel matrix is computed and stored in memory. Therefore it is completely unsuitable for any kind of large scale problem.
  - The algorithm does appear to produce correct results. While it does not match the sklearn variant 100%, results are comparable and sometimes even better. 
  - The algorithm leads to the creation of a model that is very different from the sklearn variant but functions similarly. This is proven by the different number of support vectors that each model computes.
  - The behavior of the algorithm appears to agree with that of the sklearn variant. For example, in the second trial the regularization parameter C has a very low value of 0.001, which causes a massive drop of accuracy in both variants due to the model underfitting. This holds true for both datasets / test cases.
