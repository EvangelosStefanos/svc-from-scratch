import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxopt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import Svm
import Svm_Mc
import numpy.random
import time
from functions import*

cvxopt.solvers.options['show_progress'] = False
# np.set_printoptions(precision=4, threshold=1000)


'''
X1 = np.linspace(-3, 2, num=200)
y1 = np.ones((200))*2 + rng.normal(size=200)
X2 = np.linspace(-2, 3, num=200)
y2 = np.ones((200))*(-2) + rng.normal(size=200)

X = np.append(X1, X2, axis=0)
y = np.append(y1, y2, axis=0)
XY = np.append(X.reshape((-1,1)), y.reshape((-1,1)), axis=1)

z = np.append(np.zeros(200), np.ones(200), axis=0)

model = svm(XY, z)
'''

start = time.time()
mnistTrain ='data/mnist_train.csv'
mnistTest = 'data/mnist_test.csv'
file = open('logs/res.txt', 'w')
col = ['Kernel', 'C', 'Degree', 'TrainA', 'TestA', 'TrainT']
maxSamples = 100 * 10

# mnist #

X, Y = loadMnist(mnistTrain)
X2, Y2 = loadMnist(mnistTest)
data = preprocess(X, Y, X2, Y2)
runDataset(data, maxSamples, col=col, file=file)

# gtzan #

X, Y = loadData()
X, X2, Y, Y2 = train_test_split(np.array(X), np.array(Y), test_size=0.4, random_state=0)
data = preprocess(X, Y, X2, Y2)
runDataset(data, maxSamples, col=col, file=file)


file.close()
end = time.time()
print('Running time: ' + str(end-start) + ' seconds')
