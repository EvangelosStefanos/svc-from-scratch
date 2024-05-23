import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import Svm_Mc
import numpy.random
import time


def acc(model, X, y):
  y_pred = model.predict(X)
  # print(y, y_pred)
  #yy = np.append(y, np.array(y_pred).reshape((-1, 1)), axis=1)
  #print('y - y_pred')
  #print(yy)
  matches = sum(x == y for x, y in zip(y, y_pred))
  a = matches / len(y_pred)
  return a


def loadMnist(fileName):
  mnist = pd.read_csv(fileName+'')
  y = mnist['label']
  '''
  for i in range(0, len(y)):
    if(y[i] % 2 == 0):
      y[i] = 1
    else:
      y[i] = -1
  '''
  x = mnist.drop(columns=['label'])
  return [x, y]


def loadData():
  data = pd.read_csv('data/features_3_sec.csv')
  y = data['label']
  x = data.drop(columns=['label', 'filename', 'length'])
  return [x, y]


def trainTest(dataSets, params, maxSamples=None, maxFeatures=None):
  X = dataSets['X']
  Y = dataSets['Y']
  X2 = dataSets['X2']
  Y2 = dataSets['Y2']
  kernel = params['k']
  C = params['C']
  degree = params['d']
  trainX = X[:maxSamples,:maxFeatures]
  trainY = Y[:maxSamples]
  model = Svm_Mc.Svm_Mc(kernel=kernel, C=C, degree=degree)
  # train #
  start = time.time()
  if(model.fit(trainX, trainY) == None):
    return ['None', 0, 0, 0]
  end = time.time()
  trainTime = str(end-start)
  msg = 'Train time: ' + trainTime + ' seconds\n'
  trainA = acc(model, trainX, trainY)
  msg += 'Train accuracy: ' + str(trainA) + '\n'
  # test #
  testA = acc(model, X2[:,:maxFeatures], Y2)
  msg += 'Test accuracy: ' + str(testA)
  return [msg, trainA, testA, trainTime]


def getResults(dataSets, params, maxSamples=None, maxFeatures=None, col=None, df=None):
  start = time.time()
  msgNew, trainA, testA, trainTime = trainTest(dataSets, params, maxSamples=maxSamples , maxFeatures=maxFeatures)
  end = time.time()
  print('Train-Test cycle: ', end-start, ' seconds')
  if(trainTime != None):
    trainTime = float(trainTime)
  row = pd.DataFrame({
  col[0] : [params['k']],
  col[1] : [params['C']],
  col[2] : [params['d']],
  col[3] : [trainA],
  col[4] : [testA],
  col[5] : [trainTime]
  })
  df = pd.concat([df, row])
  return df


def getParams(kernel='Linear', C=1, degree=1):
  return {'k':kernel, 'C':C, 'd':degree}


def runDataset(dataSets, maxSamples=None, maxFeatures=None, col=None, file=None):
  empty = pd.DataFrame({
  col[0] : [],
  col[1] : [],
  col[2] : [],
  col[3] : [],
  col[4] : [],
  col[5] : []
  })
  maX = 8
  df = [empty, empty, empty, empty, empty]
  for i in range(0, maX):
    C1 = (i + 1) / maX
    params = getParams('Linear', C1, 1)
    df[0] = getResults(dataSets, params, maxSamples=maxSamples, maxFeatures=maxFeatures, col=col, df=df[0])
    if(i < 4):
      C2 = pow(10, i+1)
      params = getParams('Linear', C2, 1)
      df[1] = getResults(dataSets, params, maxSamples=maxSamples, maxFeatures=maxFeatures, col=col, df=df[1])
    for j in range(0, 5):
      params = getParams('Poly', C1, j+1) # todo make it so it groups same degrees together
      df[2] = getResults(dataSets, params, maxSamples=maxSamples, maxFeatures=maxFeatures, col=col, df=df[2])
      if(i < 4):
        params = getParams('Poly', C2, j+1)
        df[3] = getResults(dataSets, params, maxSamples=maxSamples, maxFeatures=maxFeatures, col=col, df=df[3])
      print('processing', j+1, '/',4)
    print('processing', i+1, '/',maX)
  values = pd.concat([df[0], df[1], df[2], df[3], df[4]])
  pd.set_option("display.precision", 4)
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(values)
    file.write(str(values)+'\n\n')
  return


def preprocess(X, Y, X2, Y2):
  scaler = MinMaxScaler()
  scaler = scaler.fit(X)
  X = scaler.transform(X)

  numpy.random.seed(0)
  rng_state = numpy.random.get_state()
  numpy.random.shuffle(X)
  numpy.random.set_state(rng_state)
  numpy.random.shuffle(Y)

  X2 = scaler.transform(X2)
  data = {'X':X, 'Y':Y, 'X2':X2, 'Y2':Y2}
  return data

