import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import Svm_Mc

mnistTrainPath = 'input/mnist_train.csv'

mnist = pd.read_csv(mnistTrainPath+'')
target = mnist['label']
features = mnist.drop(columns=['label'])

_, features, _, target = train_test_split(features, target, test_size=1000, random_state=0, stratify=target)

scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

params = {
  'C': 1, 'kernel': 'linear'
}

modela = Svm_Mc.Svm_Mc(C=1, kernel='linear')
start = time.time()
modela.fit(features, target)
print('time:' + str(time.time()-start))
pred = modela.predict(features)

modelb = SVC(C=1, kernel='linear')
start = time.time()
modelb.fit(features, target)
print('time:' + str(time.time()-start))
true = modelb.predict(features)

print(modelb.n_support_)

accuracy = accuracy_score(true, pred)
print(accuracy)

mnistTestPath = 'input/mnist_test.csv'
mnist = pd.read_csv(mnistTestPath+'')
target = mnist['label']
features = mnist.drop(columns=['label'])

features = scaler.transform(features)

pred = modela.predict(features)
true = modelb.predict(features)
accuracy = accuracy_score(true, pred)
print(accuracy)

accuracy = accuracy_score(target, pred)
print(accuracy)
accuracy = accuracy_score(target, true)
print(accuracy)


