import numpy as np

class LogisticRegression:
  def __init__(self):
    self._weights = []
    self._learningRate = 0.02
    self._epoch = 10
    self._x = []
    self._y = []

  def linearFunction(self, data):
    return np.dot(data, self._weights)

  def logisticFunction(self, linearResult):
    return 1 / ( 1 + np.exp(-(linearResult)))

  def updateWeights(self, X, y):
    temp = np.zeros(len(X))
    for i in range(len(self._weights)):
      d = (self.logisticFunction(self.linearFunction(X)) - y) * X[i]
      temp[i] = self._weights[i] - (self._learningRate * d)

    return temp

  def logisticRegressionLoss(self, yPredicted, yActual):
    loss = -(yActual * np.log(yPredicted) + (1-yActual) * np.log(1-yPredicted))
    meanLoss = loss.mean()
    return meanLoss

  def logisticRegressionAccuracy(self, yPredicted, yActual):
    yPredicted = yPredicted.round().astype(int)
    xor = yPredicted ^ yActual
    return np.count_nonzero(xor == 0)/len(xor)

  def fit(self, X, y, learningRate, epoch):
    self._x = X
    self._y = y
    self._learningRate = learningRate
    self._epoch = epoch

    featureCount = len(self._x[0])
    self._weights = np.random.rand(featureCount)

    for epoch in range(self._epoch):
      listResult = []

      for i in range(len(self._x)):
        listResult.append(self.logisticFunction(self.linearFunction(self._x[i])))
        self._weights = self.updateWeights(self._x[i], self._y[i])

      print("epoch: ", epoch + 1,
            "\tloss: ", self.logisticRegressionLoss(np.asarray(listResult), y),
            "\taccuracy: ", self.logisticRegressionAccuracy(np.asarray(listResult), y)
      )

  def predict(self, X):
    X = np.array(X)
    if X.ndim == 2:
        listResult = []
        for i in range(len(X)):
            listResult.append(self.logisticFunction(self.linearFunction(X[i])))
        return listResult
    else:
        raise ValueError("Got array in dimension of " + str(X.ndim) + ", expecting array in a dimension of 2.")