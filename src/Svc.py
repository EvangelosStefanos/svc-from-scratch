import numpy as np
import Svm

# Support Vector Machine Multiclass Classifier (One vs All)

class Svc:


  def __init__(self, C=1.0, kernel='linear', degree=1, gamma=1):
    """
    @param C float regularization parameter (strictly positive)
    @param kernel kernel parameter one of ['linear', 'poly', 'rbf']
    @param degree if kernel is 'poly' this is the degree of the polynomial
    @param gamma rbf kernel parameter
    """
    self.kernel = kernel
    self.C = C
    self.degree = degree
    return


  def nsupport(self):
    s = '[ '
    for svm in self.svms:
      s += str(svm.nsv) + ' '
    return s + ']'


  def fit(self, x, y):
    """
    @param x array of shape (nsamples, nfeatures)
    @param y array of shape (nsamples,)
    @return fitted classifier
    """
    # process input #

    if(y.ndim != 1):
      raise Exception("ERROR: Expected array with shape (nsamples,). Got " + str(y.shape))

    self.classes = np.unique(y)
    self.nclasses = self.classes.shape[0]

    svms = [0] * self.nclasses
    class_i = np.zeros(shape=y.shape)
    # create and train one svm for each class
    for i in range(self.nclasses):
      class_i[y == self.classes[i]] = 1.0
      class_i[y != self.classes[i]] = -1.0
      svms[i] = Svm.Svm(C=self.C, kernel=self.kernel, degree=self.degree)
      svms[i].fit(x, class_i)
    self.svms = svms
    self.n_support_ = self.nsupport()
    return self


  def predict(self, x):
    """
    @param x array of shape (nsamples, nfeatures)
    @return array of shape (nsamples,)
    """
    if(np.ndim(x) != 2):
      raise Exception('ERROR: Expected array of shape (nsamples, nfeatures). Got ' + str(x.shape))
    scores = np.zeros((self.nclasses, x.shape[0])) # (nclasses, nsamples)
    for i in range(self.nclasses):
      scores[i] = self.svms[i].scoreFunction(x)
    classifyer_ids = np.argmax(scores, axis=0) # (nsamples,)
    return self.classes[classifyer_ids]


  def support(self):
    support = []
    for svm in self.svms:
      support += list(svm.sv_idx[0])
    return support

