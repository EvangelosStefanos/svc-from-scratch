import numpy as np
import cvxopt
import utils

# Support Vector Machine Binary Classifier

class Svm:  
  def __init__(self, C=1.0, kernel='linear', degree=1, gamma=1):
    """
    @param C float regularization parameter (strictly positive)
    @param kernel kernel parameter one of ['linear', 'poly', 'rbf']
    @param degree if kernel is 'poly' this is the degree of the polynomial
    @gamma rbf kernel parameter
    """
    self.KERNEL_LINEAR = 'linear'
    self.KERNEL_POLY = 'poly'
    self.KERNEL_RBF = 'rbf'
    
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-20
    cvxopt.solvers.options['reltol'] = 1e-20
    cvxopt.solvers.options['feastol'] = 1e-20

    # parameter test #

    if (C <= 0):
      raise utils.MyValueError('ERROR: Parameter C must be strictly positive. Got ' + str(C))
    self.C = C
    kernels = [self.KERNEL_LINEAR, self.KERNEL_POLY, self.KERNEL_RBF]
    if(kernel not in kernels):
      raise utils.MyValueError('ERROR: Kernel must be one of [\'linear\', \'poly\']. Got ' + str(kernel))
    self.degree = degree
    self.gamma = gamma
    if(kernel == self.KERNEL_LINEAR):
      self.kernel = self.kernel_linear
      self.scoreFunction = self.scoreFunctionLinear
    if(kernel == self.KERNEL_POLY):
      self.kernel = self.kernel_polynomial
      self.scoreFunction = self.scoreFunctionNonLinear
    if(kernel == self.KERNEL_RBF):
      self.kernel = self.kernel_rbf
      self.scoreFunction = self.scoreFunctionNonLinear
    return

  def kernel_linear(self, x_i, x_j):
    '''
    K(x_i, x_j) = x_i * (x_j)^T
    @param x_i array of shape (nsamples_i, nfeatures)
    @param x_j array of shape (nsamples_j, nfeatures)
    @return K(x_i, x_j) array of shape (nsamples_i, nsamples_j)
    '''
    return np.matmul(x_i, x_j.T)

  def kernel_polynomial(self, x_i, x_j):
    '''
    K(x_i, x_j) = (x_i * (x_j)^T + 1)^d
    @param x_i array of shape (nsamples_i, nfeatures)
    @param x_j array of shape (nsamples_j, nfeatures)
    @return K(x_i, x_j) array of shape (nsamples_i, nsamples_j)
    '''
    return (np.matmul(x_i, x_j.T) + 1.0)**self.degree

  def kernel_rbf(self, x_i, x_j):
    '''
    https://stackoverflow.com/a/50900910
    K(x_i, x_j) = exp( -gamma * norm(x_i - x_j)**2)
    @param x_i array of shape (nsamples_i, nfeatures)
    @param x_j array of shape (nsamples_j, nfeatures)
    @return K(x_i, x_j) array of shape (nsamples_i, nsamples_j)
    '''
    x_i_norm = np.linalg.norm(x_i, axis=1) ** 2
    x_j_norm = np.linalg.norm(x_j, axis=1) ** 2
    return np.exp( -self.gamma * (x_i_norm.reshape((-1, 1)) + x_j_norm.reshape((1,-1)) - 2 * np.matmul(x_i, x_j.T)))


  def compute_w(self):
    '''
    n = number of support vectors
    w = sum_i=1...n(a_i * y_i * x_i)
    @return w array of shape (nfeatures, 1)
    '''
    if(self.kernel != self.kernel_linear):
      # dont compute w for non linear kernel
      return
    # compute w only for linear kernel
    return np.sum(self.sv_x.T * self.sv_y * self.sv_a, axis=1).reshape((-1, 1)) # (nfeatures, ) -> (nfeatures, 1)


  def compute_b(self):
    '''
    n = number of support vectors
    b = sum_i=1...n(y_i - sum_j=1...n(a_j * y_j * K(x_i, x_j))) / n
    @return b float
    '''
    s = np.sum(self.sv_y - np.sum(self.sv_a * self.sv_y * self.kernel(self.sv_x, self.sv_x), axis=1), axis=0)
    return s / self.nsv


  def fit(self, x, y):
    """
    @param x array of shape (nsamples, nfeatures)
    @param y array of shape (nsamples,)
    @return fitted classifier
    """
    # process input #
    
    if(x.ndim != 2):
      raise utils.MyValueError("ERROR: Expected array with shape (nsamples, nfeatures). Got " + str(x.shape))
    if(y.ndim != 1):
      raise utils.MyValueError("ERROR: Expected array with shape (nsamples,). Got " + str(y.shape))
    if(np.setdiff1d(y, [-1, 1]).shape[0] > 0):
      raise utils.MyValueError("ERROR: Target values must be one of {-1, 1}. Got " + str(y))

    nsamples = x.shape[0]
    self.nfeatures = x.shape[1]

    # init qp vars #

    K = self.kernel(x, x)
    
    qp_P = np.outer(y, y) * K
    qp_q = np.ones((nsamples, 1)) * (-1.0)
    qp_G = np.concatenate([
      np.identity(nsamples) * (-1.0), 
      np.identity(nsamples)
    ])
    qp_h = np.concatenate([
      np.zeros((nsamples, 1)), 
      np.ones((nsamples, 1)) * self.C
    ])
    qp_A = np.reshape(y, (1, nsamples))
    qp_b = np.zeros(1)

    qp_P = cvxopt.matrix(qp_P, tc='d')
    qp_q = cvxopt.matrix(qp_q, tc='d')
    qp_G = cvxopt.matrix(qp_G, tc='d')
    qp_h = cvxopt.matrix(qp_h, tc='d')
    qp_A = cvxopt.matrix(qp_A, tc='d')
    qp_b = cvxopt.matrix(qp_b, tc='d')

    # execute qp #

    qp_res = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)
    qp_x = np.array(qp_res['x']).reshape((nsamples,)) # (nsamples, 1) -> (nsamples,)

    # get support vector data #

    sv_mask = (1e-10 <= qp_x) & (qp_x <= self.C)
    sv_idx = np.asarray(sv_mask).nonzero()
    self.sv_idx = sv_idx

    if(qp_x[sv_idx].shape[0] == 0):
      raise utils.MyValueError('ERROR: No support vectors found. Got ' + str(qp_x))

    self.sv_a = qp_x[sv_idx]
    self.sv_x = x[sv_idx]
    self.sv_y = y[sv_idx]
    self.nsv = len(self.sv_a)
    
    self.sv_w = self.compute_w()
    self.sv_b = self.compute_b()
    return self


  def scoreFunctionLinear(self, x):
    '''
    f(x*) = (x*)^T * w + b
    @param x array of shape (nsamples, nfeatures)
    @return f(x*) array of shape (nsamples,)
    '''
    # for linear kernel use w for prediction
    # x(nsamples, nfeatures) * w(nfeatures, 1) -> xw(nsamples, 1) -> y(nsamples,)
    return np.reshape((np.matmul(x, self.sv_w)) + self.sv_b, (-1))


  def scoreFunctionNonLinear(self, x):
    '''
    f(x*) = sum_i=1...n(a_i * y_i * K(x_i, x*)) + b
    @param x array of shape (nsamples, nfeatures)
    @return f(x*) array of shape (nsamples,)
    '''
    # for non linear kernel don't use w for prediction
    return np.sum(self.kernel(x, self.sv_x) * self.sv_y * self.sv_a, axis=1) + self.sv_b


  def predict(self, x):
    '''
    g(x) = sign(f(x))
    @param x array of shape (nsamples, nfeatures)
    @return g(x) array of shape (nsamples,)
    '''
    if(np.ndim(x) != 2):
      raise utils.MyValueError('ERROR: Expected array of shape (nsamples, nfeatures). Got ' + str(x.shape))
    return np.sign(self.scoreFunction(x))
