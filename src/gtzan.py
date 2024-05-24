import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trial import trial

# test case for the gtzan dataset #

LOGPATH = 'logs/gtzan.csv'
DATANAME = 'gtzan'
DATAPATH = 'input/features_3_sec.csv'

gtzan = pd.read_csv(DATAPATH+'')
target = gtzan['label']
features = gtzan.drop(columns=['label', 'filename', 'length'])

# use a small sample for training
featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target, train_size=1000, random_state=0, stratify=target)

scaler = MinMaxScaler()
scaler.fit(featuresTrain)
featuresTrain = scaler.transform(featuresTrain)
featuresTest = scaler.transform(featuresTest)
    
stats = []
stats += trial(1, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(0.001, 'linear', 1, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(1, 'poly', 2, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
stats += trial(0.001, 'poly', 5, featuresTrain, targetTrain, featuresTest, targetTest, DATANAME)
pd.DataFrame(stats).to_csv(LOGPATH, mode='w')
