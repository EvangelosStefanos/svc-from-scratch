import numpy as np
import cvxopt

class Svm:  
  def __init__(self, C=1.0, kernel='linear', degree=1):
    """
    @param C float regularization parameter (strictly positive)
    @param kernel kernel parameter one of ['linear', 'poly']
    @param degree if kernel is 'poly' this is the degree of the polynomial
    """
    self.KERNEL_LINEAR = 'linear'
    self.KERNEL_POLY = 'poly'
    
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 1000

    # parameter test #

    if (C <= 0):
      raise Exception('ERROR: Parameter C must be strictly positive. Got ' + str(C))
    self.C = C
    kernels = [self.KERNEL_LINEAR, self.KERNEL_POLY]
    if(kernel not in kernels):
      raise Exception('ERROR: Kernel must be one of [\'linear\', \'poly\']. Got ' + str(kernel))
    if(kernel == self.KERNEL_LINEAR):
      self.kernel = self.kernel_linear
      self.scoreFunction = self.scoreFunctionLinear
    if(kernel == self.KERNEL_POLY):
      self.kernel = self.kernel_polynomial
      self.degree = degree
      self.scoreFunction = self.scoreFunctionNonLinear
    return


  def kernel_linear(self, x_i, x_j):
    '''
    K(x_i, x_j) = (x_i)^T * x_j
    @param x_i float
    @param x_j float
    @return K(x_i, x_j) float
    '''
    return np.inner(x_i, x_j)


  def kernel_polynomial(self, x_i, x_j):
    '''
    K(x_i, x_j) = ((x_i)^T * x_j + 1)^d
    @param x_i float
    @param x_j float
    @return K(x_i, x_j) float
    '''
    return pow((np.inner(x_i, x_j) + 1), self.degree)


  def compute_w(self):
    '''
    n = number of support vectors
    w = sum_i=1...n(a_i * y_i * x_i)
    w array of shape (nfeatures, 1)
    '''
    if(self.kernel != self.kernel_linear):
      # dont compute w for non linear kernel
      return
    # compute w only for linear kernel
    sv_w = 0
    for i in range(self.nsv):
      # (nfeatures,) = (1) * (1) * (nfeatures,)
      sv_w += self.sv_a[i] * self.sv_y[i] * self.sv_x[i]
    self.sv_w = sv_w.reshape((sv_w.shape[0], 1)) # (nfeatures, ) -> (nfeatures, 1)
    return


  def compute_b(self):
    '''
    n = number of support vectors
    b = sum_i=1...n(y_i - sum_j=1...n(a_j * y_j * K(x_i, x_j))) / n
    b float
    '''
    sum_outer = 0
    for i in range(0, self.nsv):
      sum_inner = 0
      for j in range(0, self.nsv):
        sum_inner += self.sv_a[j] * self.sv_y[j] * self.kernel(self.sv_x[i], self.sv_x[j])
      sum_outer += self.sv_y[i] - sum_inner
    self.sv_b = sum_outer / self.nsv
    return


  def fit(self, x, y):
    """
    @param x array of shape (nsamples, nfeatures)
    @param y array of shape (nsamples,)
    @return fitted classifier
    """
    # process input #
    
    if(x.ndim != 2):
      raise Exception("ERROR: Expected array with shape (nsamples, nfeatures). Got " + str(x.shape))
    if(y.ndim != 1):
      raise Exception("ERROR: Expected array with shape (nsamples,). Got " + str(y.shape))
    y = np.reshape(y, (y.shape[0], 1))
    if(np.setdiff1d(y, [-1, 1]).shape[0] > 0):
      raise Exception("ERROR: Target values must be one of {-1, 1}. Got " + str(y))

    nsamples = x.shape[0]
    self.nfeatures = x.shape[1]

    # init qp vars #

    qp_P = np.zeros(shape=(nsamples, nsamples))
    for i in range(nsamples):
      for j in range(nsamples):
        qp_P[i,j] = y[i] * y[j] * self.kernel(x[i], x[j])

    qp_q = np.ones(shape=(nsamples, 1)) * (-1)
    I = np.identity(nsamples)
    qp_G = np.concatenate([I * (-1), I])
    qp_h_1 = np.zeros(shape=(nsamples, 1))
    qp_h_2 = np.ones(shape=(nsamples, 1)) * self.C
    qp_h = np.concatenate([qp_h_1, qp_h_2])
    qp_A = y.reshape((1, nsamples))
    qp_b = 0

    qp_P = cvxopt.matrix(qp_P, tc='d')
    qp_q = cvxopt.matrix(qp_q, tc='d')
    qp_G = cvxopt.matrix(qp_G, tc='d')
    qp_h = cvxopt.matrix(qp_h, tc='d')
    qp_A = cvxopt.matrix(qp_A, tc='d')
    qp_b = cvxopt.matrix(qp_b, tc='d')

    # execute qp #

    qp_res = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)
    qp_x = np.array(qp_res['x'])

    # get support vector data #

    sv_idx = np.where(qp_x > 1e-3)[0]

    if(qp_x[sv_idx].shape[0] == 0):
      raise Exception('ERROR: No support vectors found. Got ' + str(qp_x))

    print(str(qp_x[sv_idx].shape))
    self.sv_a = qp_x[sv_idx].reshape(-1)
    self.sv_x = x[sv_idx]
    self.sv_y = y[sv_idx].reshape(-1)
    self.nsv = len(self.sv_a)
    
    self.compute_w()
    self.compute_b()
    return self


  def scoreFunctionLinear(self, x):
    '''
    f(x*) = (x*)^T * w + b
    @param x array of shape (nsamples, nfeatures)
    @return f(x*) array of shape (nsamples,)
    '''
    # for linear kernel use w for prediction
    # x (1, nfeatures) | w (nfeatures, 1) | b ()
    # (1, 1) = (1, nfeatures) * (nfeatures, 1) + ()
    # x(nsamples, nfeatures) * w(nfeatures, 1) -> xw(nsamples, 1) -> y(nsamples,)
    return np.reshape((np.matmul(x, self.sv_w)) + self.sv_b, (-1))


  def scoreFunctionNonLinear(self, x):
    '''
    f(x*) = sum_i=1...n(a_i * y_i * K(x_i, x*)) + b
    @param x array of shape (nsamples, nfeatures)
    @return f(x*) array of shape (nsamples,)
    '''
    # for non linear kernel don't use w for prediction
    f = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
      sum_ = 0
      for j in range(self.nsv):
        #() = () * ()
        sum_ += self.sv_a[j] * self.sv_y[j] * self.kernel(x[i], self.sv_x[j])
      f[i] = sum_ + self.sv_b
    return f


  def predict(self, x):
    '''
    g(x) = sign(f(x))
    @param x array of shape (nsamples, nfeatures)
    @return g(x) array of shape (nsamples,)
    '''
    if(np.ndim(x) != 2):
      raise Exception('ERROR: Expected array of shape (nsamples, nfeatures). Got ' + str(x.shape))
    return np.sign(self.scoreFunction(x))