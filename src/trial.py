import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from Svc import Svc

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
    accx = accuracy_score(self.y, predx)
    predxn = m.predict(self.xn)
    accxn = accuracy_score(self.yn, predxn)
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
    m1 = self.evaluate('svc-from-scratch', Svc, p)
    r1 = self.create_record(p, m1)
    m2 = self.evaluate('sklearn', SVC, p)
    r2 = self.create_record(p, m2)
    return [r1, r2]


  def trials(self):
    stats = []
    for p in iter(ParameterGrid(self.grid)):
      stats += self.trial(p)
    return stats

