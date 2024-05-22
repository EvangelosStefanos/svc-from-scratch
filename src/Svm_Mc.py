import numpy as np
import sys
import Svm as SVM
from functions import*

class Svm_Mc:
  # One vs All

  def __init__(self, kernel='lin', C=1.0, degree=1):
    self.kernel = kernel
    self.C = C
    self.degree = degree
    return


  def to2D(self, data, name):
    shape = np.shape(data)
    ndims = len(shape)
    if(ndims < 1):
      print('parameter ', name, ' must have at least one dimension')
      sys.exit()
    elif(ndims == 1):
      rows = shape[0]
      cols = 1
      data = np.array(data).reshape((rows, cols))
    return data


  def validate_y(self, y):
    # y must be a 2d array of size (nsamples, 1)
    valid = True
    if(len(y.shape) != 2):
      print('y must be a 2d array')
      valid = False
    else:
      if(y.shape[1] != 1):
        print('second dimension of y must be 1')
        valid = False
    return valid


  def createTargetClassLabels(self, target_label, y):
    # target_label must be an integer
    # y must be a 2d array of size (nsamples, 1)
    if(not self.validate_y):
      sys.exit()
    yt = np.copy(y)
    for i in range(0, y.shape[0]):
      if(y[i, 0] == target_label):
        yt[i, 0] = 1
      else:
        yt[i, 0] = -1
    return yt


  def createLabelDict(self, y):
    # y must be a 2d array of size (nsamples, 1)
    if(not self.validate_y):
      sys.exit()
    classLabels = dict()
    for i in range(0, y.shape[0]):
      if(y[i, 0] not in classLabels.values()):
        label_id = len(classLabels)
        classLabels[label_id] = y[i, 0]
    self.classLabels = classLabels
    return


  def fit(self, x, y):

    # process input #

    x = self.to2D(x, 'x') # shape=(nsamples, nfeatures)
    y = self.to2D(y, 'y') # shape=(nsamples, 1)

    self.createLabelDict(y)

    nsamples = y.shape[0]

    y_c = dict()
    if(len(self.classLabels) == 2):
      print('running multiclass svm on binary dataset is not supported. use binary svm instead. exiting')
      sys.exit()
    # if number of classes is n create n datasets
    for i in range(0, len(self.classLabels)):
      y_c[i] = self.createTargetClassLabels(self.classLabels[i], y)
    self.y_c = y_c
    svms = [0] * len(self.classLabels)
    for i in range(0, len(self.classLabels)):
      yt = y_c[i]
      #print(np.append(y, yt, axis=1))
      svms[i] = SVM.Svm(kernel=self.kernel, C=self.C, degree=self.degree)
      if(svms[i].fit(x, yt) == None):
        return None
      # print('Classifyer Trained: ', i+1, '/', len(self.classLabels))
    self.svms = svms
    return self


  def predict(self, x):
    # x must be 2d array (nsamples, nfeatures)
    if(len(np.shape(x)) != 2):
      print('x must be a 2d array exiting')
      sys.exit()
    scores = [0] * len(self.classLabels)
    for i in range(0, len(self.classLabels)):
      scores[i] = self.svms[i].scoreFunction(x).ravel()
    scores = np.array(scores)
    classifyer_ids = np.argmax(scores, axis=0)
    y = [0] * np.shape(x)[0]
    for i in range(0, len(classifyer_ids)):
      y[i] = self.classLabels[classifyer_ids[i]]
    '''
    print(scores)
    print('y - classifyer_ids')
    print(np.append(np.array(y).reshape((-1, 1)),
    np.array(classifyer_ids).reshape((-1, 1)), axis=1))
    print(self.classLabels)
    '''
    return np.array(y)
