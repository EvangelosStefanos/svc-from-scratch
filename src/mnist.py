import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import Svc

LOGPATH = 'logs/mnist.csv'

def write(msg):
  file = open(LOGPATH, 'a')
  file.write(msg)
  file.close()
  return

file = open(LOGPATH, 'w')
file.close()

# train #
mnistTrainPath = 'input/mnist_train.csv'
mnist = pd.read_csv(mnistTrainPath+'')
target = mnist['label']
features = mnist.drop(columns=['label'])

# use a small sample for training
features, _, target, _ = train_test_split(features, target, train_size=100, random_state=0, stratify=target)

scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
# train #

# test #
mnistTestPath = 'input/mnist_test.csv'
mnist2 = pd.read_csv(mnistTestPath+'')
target2 = mnist2['label']
features2 = mnist2.drop(columns=['label'])
features2 = scaler.transform(features2)
# test #

write('Dataset (train size / test size / features):,mnist (' + str(features.shape[0]) + ' / ' + str(features2.shape[0]) + ' / ' + str(features.shape[1]) + ')\n')

def trial(C, kernel, degree, features, target, features2, target2):
  write('Parameters:,C = ' + str(C) + ' / kernel = ' + str(kernel) + ' / degree = ' + str(degree) + '\n')
  model1 = Svc.Svc(C=C, kernel=kernel, degree=degree)
  start = time.time()
  model1.fit(features, target)
  time1 = time.time()-start
  pred = model1.predict(features)

  model2 = SVC(C=C, kernel=kernel, degree=degree)
  start = time.time()
  model2.fit(features, target)
  time2 = time.time()-start
  write('Fitting time (scv-from-scratch / sklearn):,' + str(round(time1, 4)) + ' / ' + str(round(time2, 4)) + '\n')
  write('Number of support vectors (svc-from-scratch / sklearn):,' + str(model1.nsupport()) + ' / ' + str(model2.n_support_) + '\n')
  true = model2.predict(features)
  accuracy1 = accuracy_score(true, pred)

  pred = model1.predict(features2)
  true = model2.predict(features2)
  accuracy2 = accuracy_score(true, pred)
  write('Fraction of predictions matching sklearn (train / test):,' + str(accuracy1) + ' / ' + str(accuracy2) + '\n')

  accuracy3 = accuracy_score(target2, pred)
  accuracy4 = accuracy_score(target2, true)
  write('Fraction of correct predictions (svc-from-scratch / sklearn):,' + str(accuracy3) + ' / ' + str(accuracy4) + '\n')
  return

trial(1, 'linear', 1, features, target, features2, target2)
trial(0.01, 'linear', 1, features, target, features2, target2)
trial(1, 'poly', 2, features, target, features2, target2)
trial(1, 'poly', 3, features, target, features2, target2)
