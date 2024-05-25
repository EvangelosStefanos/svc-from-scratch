import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trial import trial
import time

# test case for the mnist dataset #

start = time.time()
LOGPATH = 'logs/mnist.csv'
DATANAME = 'mnist'
TRAINPATH = 'input/mnist_train.csv'
TESTPATH = 'input/mnist_test.csv'

# train #
mnist = pd.read_csv(TRAINPATH+'')
targetTrain = mnist['label']
featuresTrain = mnist.drop(columns=['label'])

# use a small sample for training
featuresTrain, _, targetTrain, _ = train_test_split(featuresTrain, targetTrain, train_size=100, random_state=0, stratify=targetTrain)

scaler = MinMaxScaler()
scaler.fit(featuresTrain)
featuresTrain = scaler.transform(featuresTrain)
# train #

# test #
mnist2 = pd.read_csv(TESTPATH+'')
targetTest = mnist2['label']
featuresTest = mnist2.drop(columns=['label'])
featuresTest = scaler.transform(featuresTest)
# test #

stats = []
stats += trial(0.0001, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(0.01, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(1, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(100, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(10000, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(1, 'poly', 2, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(1, 'poly', 3, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(1, 'poly', 4, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
pd.DataFrame(stats, index=pd.RangeIndex(start=1, stop=len(stats)+1, name='Trial')).to_csv(LOGPATH, mode='w')
print('Running time: ', time.time()-start)
