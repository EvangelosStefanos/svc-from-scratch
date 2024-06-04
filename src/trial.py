import time
import sklearn.model_selection
import sklearn.metrics
import sklearn.svm
import utils
import svc


class Trial:


  def __init__(self, dataname, x, y, xn, yn, grid):
    self.dataname = dataname
    self.x = x
    self.y = y
    self.xn = xn
    self.yn = yn
    self.grid = grid
    return


  def create_record(self, p, m):
    r = {}
    r['Dataset'] = self.dataname
    r['nTrain'] = self.x.shape[0]
    r['nFeatures'] = self.x.shape[1]
    r['nTest'] = self.xn.shape[0]
    r['Model'] = m['name']
    r['Time'] = round(m['time'], 4)
    r['Accuracy (train)'] = round(m['accx'], 4)
    r['Accuracy (test)'] = round(m['accxn'], 4)
    r = {**r, **p}
    r['Number of support vectors'] = str(m['model'].n_support_)
    return r


  def evaluate(self, name, model, p):
    start = time.time()
    m = model(**p)
    m.fit(self.x, self.y)

    predx = m.predict(self.x)
    accx = sklearn.metrics.accuracy_score(self.y, predx)
    predxn = m.predict(self.xn)
    accxn = sklearn.metrics.accuracy_score(self.yn, predxn)
    t = time.time()-start
    return {
      'name':name,
      'model':m, 
      'time':t, 
      'predx':predx,
      'accx':accx,
      'predxn':predxn,
      'accxn':accxn,
      'nsupp':m.n_support_
      }


  def trial(self, p):
    m1 = self.evaluate('svc-from-scratch', svc.Svc, p)
    r1 = self.create_record(p, m1)
    m2 = self.evaluate('sklearn', sklearn.svm.SVC, p)
    r2 = self.create_record(p, m2)
    return [r1, r2]


  def trials(self):
    stats = []
    for i, p in enumerate(sklearn.model_selection.ParameterGrid(self.grid)):
      try:
        print(i, ' : ', p)
        stats += self.trial(p)
      except utils.MyValueError as e:
        print(e)
        continue
    return stats

