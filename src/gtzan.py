import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trial import trial
import time

# test case for the gtzan dataset #

start = time.time()
LOGPATH = 'logs/gtzan.csv'
DATANAME = 'gtzan'
DATAPATH = 'input/features_3_sec.csv'

gtzan = pd.read_csv(DATAPATH+'')
target = gtzan['label']
features = gtzan.drop(columns=['label', 'filename', 'length'])

# use a small sample for training
featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(
  features, target, train_size=1000, 
  random_state=0, stratify=target
)

scaler = MinMaxScaler()
scaler.fit(featuresTrain)
featuresTrain = scaler.transform(featuresTrain)
featuresTest = scaler.transform(featuresTest)
    
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
