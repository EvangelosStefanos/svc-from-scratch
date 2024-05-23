import numpy as np
import Svm as SVM

class Svm_Mc:
  # One vs All

  def __init__(self, C=1.0, kernel='linear', degree=1):
    self.kernel = kernel
    self.C = C
    self.degree = degree
    return


  def fit(self, x, y):
    """
    @param x array of shape (nsamples, nfeatures)
    @param y array of shape (nsamples,)
    @return fitted classifier
    """
    # process input #

    if(y.ndim != 1):
      raise Exception("ERROR: Expected array with shape (nsamples,). Got " + str(y.shape))

    classes = np.unique(y)
    self.classes = classes
    nclasses = classes.shape[0]
    self.nclasses = nclasses

    svms = [0] * nclasses
    class_i = np.copy(y)
    # create and train one svm for each class
    for i in range(nclasses):
      class_i[y == classes[i]] = 1
      class_i[y != classes[i]] = -1
      svms[i] = SVM.Svm(C=self.C, kernel=self.kernel, degree=self.degree)
      svms[i].fit(x, class_i)
    self.svms = svms
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
