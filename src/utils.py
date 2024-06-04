class MyValueError(ValueError):
  pass


def createGrid():
  return [
    {
      'C':[1e-3, 1, 1e3],
      'kernel':['linear'],
      'degree':[1],
      'gamma':[1],
    },
    {
      'C':[1],
      'kernel':['poly'],
      'degree':[2, 3, 4],
      'gamma':[1],
    },
    {
      'C':[1],
      'kernel':['rbf'],
      'degree':[1],
      'gamma':[1e-3, 1, 1e3],
    }
  ]

