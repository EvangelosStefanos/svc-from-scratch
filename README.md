# svc-from-scratch
  Support Vector Machine Classifier from scratch

## Details
  An implementation of a support vector machine classifier written in python for multiclass classification problems. Internally the [cvxopt](https://cvxopt.org/) package is used to solve the quadratic programming optimization problem. Support for the multiclass case is provided by the one-vs-all strategy.

## Evaluation
  The implementation is tested against the [SVC class from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). There are two test cases implemented, one for the mnist dataset ([mnist_train.csv](https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv) / [mnist_test.csv](https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_test.csv)) and one for the gtzan dataset ([features_3_sec.csv](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)). Each test case contains eight trials, each with different parameter combinations. The results can be found in [mnist.csv](logs/mnist.csv) and [gtzan.csv](logs/gtzan.csv).

## Observations
  - The algorithm is significantly slower and does not scale well at all. Unsurprising, considering the fact that a N x N (where N is the number of training samples) kernel matrix is computed and stored in memory. Therefore it is completely unsuitable for any kind of large scale problem.
  - The algorithm does appear to produce correct results. While it does not match the sklearn variant 100%, results are comparable and sometimes even better. 
  - The difference in the number of support vectors as well as the set of support vectors suggest that the algorithm leads to the creation of a model that is different from the sklearn variant. Despite that, the model does function in a similar manner.
  - The behavior of the algorithm appears to agree with that of the sklearn variant. For example, in the first trial the regularization parameter C has a value very close to zero, which causes a massive drop of accuracy in both variants due to the models underfitting. This holds true for both datasets / test cases.

## Conclusions
  The algorithm is unsuitable for any kind of use. It has become obsolete in the recent years due to its bad performance and the existance of much better algorithms ([SMO](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)).
