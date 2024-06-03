import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trial import Trial
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
featuresTrain, _, targetTrain, _ = train_test_split(
  featuresTrain, targetTrain, train_size=1000, 
  random_state=0, stratify=targetTrain
)

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


grid = [
  {
    'C':[1e-4, 1e-2, 1, 1e2, 1e4],
    'kernel':['linear'],
    'degree':[1],
    'gamma':[1],
  },
  {
    'C':[1],
    'kernel':['poly'],
    'degree':[2, 3, 4],
    'gamma':[1],
  },
  {
    'C':[1],
    'kernel':['rbf'],
    'degree':[1],
    'gamma':[1e-4, 1e-2, 1, 1e2, 1e4],
  }
]

trial = Trial(DATANAME, featuresTrain, targetTrain, featuresTest, targetTest, grid)
stats = trial.trials()
pd.DataFrame(stats, index=pd.RangeIndex(start=1, stop=len(stats)+1, name='Trial')).to_csv(LOGPATH, mode='w')
print('Running time: ', time.time()-start)
