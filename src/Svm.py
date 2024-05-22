import numpy as np
import cvxopt
import sys
import matplotlib.pyplot as plt
from functions import*

class Svm:

  def __init__(self, kernel='Linear', C=1.0, degree=1):

    # parameter test #

    if (C <= 0):
      print('Parameter C must be strictly positive. exiting')
      sys.exit()
    self.C = C
    kernels = ['Linear', 'Poly']
    if(kernel not in kernels):
      print('Kernel is not supported.')
      return
    if(kernel == 'Linear'):
      self.kernel = self.kernel_linear
      self.scoreFunction = self.scoreFunctionLinear
    if(kernel == 'Poly'):
      self.kernel = self.kernel_polynomial
      self.scoreFunction = self.scoreFunctionNonLinear
      self.degree = degree
    return


  def kernel_linear(self, x_i, x_j):
    '''
    K(x_i, x_j) = (x_i)^T * x_j
    '''
    if(naxes(x_i) != 1 or naxes(x_i) != 1):
      print('x_i and x_j must be vectors. exiting')
      sys.exit()
    return np.inner(x_i, x_j);


  def kernel_polynomial(self, x_i, x_j):
    '''
    K(x_i, x_j) = ((x_i)^T * x_j + 1)^d
    '''
    return pow((np.inner(x_i, x_j) + 1), self.degree)


  def to2D(self, data, name):
    if(naxes(data) < 1):
      print('parameter ', name, ' must have at least one dimension')
      sys.exit()
    elif(naxes(data) == 1):
      data = np.array(data).reshape((len(data), 1)) # (nfeatures, ) -> (nfeatures, 1)
    return data


  def validate_y(self, y):
    valid = True
    if(naxes(y) != 2):
      print('y must be a 2d array')
      valid = False
    else:
      if(y.shape[1] != 1):
        print('second dimension of y must be 1')
        valid = False
    return valid


  def createTargetClassLabels(self, y):
    # y must be a 2d array of size (nsamples, 1)
    if(not self.validate_y(y)):
      sys.exit()
    yt = copy(y)
    for i in range(0, y.shape[0]):
      yt[i, 0] = self.mapf[y[i, 0]]
    return yt


  def createCorrectLabels(self, y):
    # y must be a 2d array of size (nsamples, 1)
    if(not self.validate_y(y)):
      sys.exit()
    valid = [1, -1]
    mapf = dict()
    mapb = dict()
    for i in range(0, y.shape[0]):
      if(y[i, 0] not in mapf.keys()):
        if(len(mapf) == 2):
          print('binary svm got != 2 class labels. exiting', y)
          sys.exit()
        v = valid[len(mapf)]
        mapf[y[i, 0]] = v
        mapb[v] = y[i, 0]
    self.mapf = mapf
    self.mapb = mapb
    # if number of classes is 2 create 1 dataset
    return self.createTargetClassLabels(y)


  def compute_w(self):
    '''
    n = number of support vectors
    w = sum_i=1...n(a_i, y_i, x_i)
    '''
    if(self.kernel != self.kernel_linear):
      # dont compute w for non linear kernel
      return
    # compute w only for linear kernel
    sv_w = 0
    for i in range(0, self.nsv):
      # (nfeatures,) = (1) * (1) * (nfeatures,)
      sv_w += (self.sv_a[i] * self.sv_y[i]) * self.sv_x[i]
    sv_w = sv_w.reshape((sv_w.shape[0], 1)) # (nfeatures, ) -> (nfeatures, 1)
    self.sv_w = sv_w
    #print('w: \n', self.sv_w)
    return


  def compute_b(self):
    '''
    n = number of support vectors
    b = sum_i=1...n(y_i - sum_j=1...n(a_j * y_j * K(x_i, x_j))) / n
    '''
    sum_outer = 0
    for i in range(0, self.nsv):
      sum_inner = 0
      for j in range(0, self.nsv):
        sum_inner += self.sv_a[j] * self.sv_y[j] * self.kernel(self.sv_x[i], self.sv_x[j])
      sum_outer += self.sv_y[i] - sum_inner

    sv_b = sum_outer / self.nsv
    self.sv_b = sv_b
    #print('b: ', self.sv_b)
    return


  def fit(self, x, y):

    # process input #

    x = self.to2D(x, 'x') # shape=(nsamples, nfeatures)
    y = self.to2D(y, 'y') # shape=(nsamples, 1)

    y1=copy(y)

    #y = self.createCorrectLabels(y)
    # print(np.append(x, np.append(y2, y1, axis=1), axis=1))

    y2=copy(y)
    y3=np.append(y1,y2,axis=1)
    #print(y3)
    nsamples = x.shape[0]
    self.nfeatures = x.shape[1]

    # init qp vars #

    qp_P = np.ndarray(shape=(nsamples, nsamples))
    for i in range(0, nsamples):
      for j in range(0, nsamples):
        qp_P[i,j] = y[i] * y[j] * self.kernel(x[i], x[j])

    qp_q = np.ones(shape=(nsamples, 1)) * (-1)
    I = np.identity(nsamples)
    qp_G = np.concatenate([I * (-1), I])
    qp_h_1 = np.zeros(shape=(nsamples,1))
    qp_h_2 = np.ones(shape=(nsamples,1)) * self.C
    qp_h = np.concatenate([qp_h_1, qp_h_2])
    qp_A = np.array(y).reshape((1,nsamples))
    qp_b = 0

    self.qp = dict()
    self.qp['P'] = qp_P
    self.qp['q'] = qp_q
    self.qp['G'] = qp_G
    self.qp['h_1'] = qp_h_1
    self.qp['h_2'] = qp_h_2
    self.qp['h'] = qp_h
    self.qp['A'] = qp_A
    self.qp['b'] = qp_b

    qp_P = cvxopt.matrix(qp_P, tc='d')
    qp_q = cvxopt.matrix(qp_q, tc='d')
    qp_G = cvxopt.matrix(qp_G, tc='d')
    qp_h = cvxopt.matrix(qp_h, tc='d')
    qp_A = cvxopt.matrix(np.array(qp_A, dtype=np.double), tc='d')
    qp_b = cvxopt.matrix(qp_b, tc='d')

    # execute qp #

    qp_res = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)
    qp_x = np.array(qp_res['x'])
    self.qp['x'] = qp_x

    # get support vector data #

    sv_idx = np.where(qp_x > 1e-5)[0]
    sv_a = qp_x[sv_idx]
    nsv = len(sv_a)
    if(nsv == 0):
      print('no support vectors. exiting')
      return None
    sv_x = x[sv_idx]
    sv_y = y[sv_idx]

    self.sv_a = sv_a.ravel()
    self.nsv = nsv
    self.sv_x = sv_x
    self.sv_y = sv_y.ravel()

    self.compute_w()
    self.compute_b()
    return self


  def printShape(*argv):
    for i in argv:
      print(np.shape(i))
    return


  def scoreFunctionLinear(self, x):
    '''
    f_x* = (x*)^T * w + b
    '''
    # for linear kernel use w for prediction
    # x (1, nfeatures) | w (nfeatures, 1) | b ()
    # (1, 1) = (1, nfeatures) * (nfeatures, 1) + ()
    if(naxes(x) != 2):
      print('x must be a 2d array. exiting')
      sys.exit()
    # (nsamples, 1) = (nsamples, nfeatures) * (nfeatures, 1) + ()
    y = (np.matmul(x, self.sv_w)) + self.sv_b
    return y


  def scoreFunctionNonLinear(self, x):
    '''

    f_x* = sum_i=1...n(a_i * y_i * K(x_i, x*)) + b
    '''
    # for non linear kernel don't use w for prediction
    if(naxes(x) != 2):
      print('scoreFunctionNonLinear x must be a 2d array. exiting')
      sys.exit()
    y = [0] * x.shape[0]
    for i in range(0, x.shape[0]):
      sum_ = 0
      for j in range(0, self.nsv):
        #() = () * () * ()
        sum_ += self.sv_a[j] * self.sv_y[j] * self.kernel(x[i], self.sv_x[j])
      y[i] = sum_ + self.sv_b
    return np.array(y).reshape((-1,1))


  def predict(self, x):
    '''
    g_x = sign(f_x)
    '''
    # x must be 2d array
    if(naxes(x) != 2):
      print('predict x must be a 2d array. exiting')
      sys.exit()
    out = [0] * len(x)
    for i in range(0, len(x)):
      sign = np.sign(self.scoreFunction(x))[i, 0]
      #out[i] = self.mapb[sign]
      out[i] = sign
    return np.array(out)
