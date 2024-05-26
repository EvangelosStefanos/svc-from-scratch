import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from Svc import Svc

# train-test-evaluate cycle #

def trial(C, kernel, degree, featuresTrain, targetTrain, featuresTest, targetTest, dataname):
  model1 = Svc(C=C, kernel=kernel, degree=degree) # svc-from-scratch #
  start = time.time()
  model1.fit(featuresTrain, targetTrain)
  time1 = time.time()-start

  predTrain = model1.predict(featuresTrain)
  predTest = model1.predict(featuresTest)
  accuracyTrain = accuracy_score(targetTrain, predTrain)
  accuracyTest = accuracy_score(targetTest, predTest)

  model2 = SVC(C=C, kernel=kernel, degree=degree) # sklearn.svm.SVC #
  start = time.time()
  model2.fit(featuresTrain, targetTrain)
  time2 = time.time()-start

  predTrain2 = model2.predict(featuresTrain)
  predTest2 = model2.predict(featuresTest)
  accuracyTrain2 = accuracy_score(targetTrain, predTrain2)
  accuracyTest2 = accuracy_score(targetTest, predTest2)

  matchTrain = accuracy_score(predTrain2, predTrain)
  matchTest = accuracy_score(predTest2, predTest)

  s1 = set(model1.support())
  s2 = set(model2.support_)
  disjoint = s1.isdisjoint(s2)
  m1subset = False
  m2subset = False
  if not disjoint:
    m1subset = s1.issubset(s2)
    m2subset = s2.issubset(s1)

  return [
    { 
      'Dataset':dataname, 
      'nTrain':featuresTrain.shape[0], 
      'nTest':featuresTest.shape[0], 
      'nfeats':featuresTrain.shape[1], 
      'C':C, 'kernel':kernel, 'degree':degree, 
      'Fit time svc-from-scratch':round(time1, 4), 
      'Fit time sklearn':round(time2, 4), 
      'Accuracy svm-from-scratch (train)':round(accuracyTrain, 4), 
      'Accuracy svm-from-scratch (test)':round(accuracyTest, 4), 
      'Accuracy sklearn (train)':round(accuracyTrain2, 4), 
      'Accuracy sklearn (test)':round(accuracyTest2, 4), 
      'Fraction of predictions matching sklearn (train)':round(matchTrain, 4), 
      'Fraction of predictions matching sklearn (test)':round(matchTest, 4), 
      'Number of support vectors svm-from-scratch':str(model1.nsupport()),
      'Number of support vectors sklearn':str(model2.n_support_),
      'Disjoint':disjoint,
      'svc-from-scratch subset':m1subset,
      'sklearn subset':m2subset,
    },
  ]
