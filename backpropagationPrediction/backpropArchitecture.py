import numpy as np 

class NeuralNetwork(object):
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes     = inputNodes
        self.hiddenNodes    = hiddenNodes
        self.outputNodes    = outputNodes

        self.weights_input_hidden   = np.random.normal(0.0, self.inputNodes**-0.5, 
                                                    (self.inputNodes, self.hiddenNodes))
        self.weights_hidden_output  = np.random.normal(0.0, self.hiddenNodes**-0.5, 
                                                    (self.hiddenNodes, self.outputNodes))

        self.learningRate = learningRate
        self.activationFunction = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        nRecords = features.shape[0]
        deltaW_input_hidden     = np.zeros(self.weights_input_hidden.shape)
        deltaW_hidden_output    = np.zeros(self.weights_hidden_output.shape) 

        for X, y in zip(features, targets):
            finalOutputs, hiddenOutputs = self.forwardPassTrain(X)
            deltaW_input_hidden, deltaW_hidden_output = self.backpropagation(finalOutputs, hiddenOutputs, 
                                                                            X, y, deltaW_input_hidden, deltaW_hidden_output)

        self.updateWeights(deltaW_input_hidden, deltaW_hidden_output, nRecords)

    def forwardPassTrain(self, X):
        hiddenInputs    = np.dot(X, self.weights_input_hidden)
        hiddenOutputs   = self.activationFunction(hiddenInputs)

        finalInputs     = np.dot(hiddenOutputs, self.weights_hidden_output)
        finalOutputs    = finalInputs

        return finalOutputs, hiddenOutputs

    def backpropagation(self, finalOutputs, hiddenOutputs, X, y, deltaW_input_hidden, deltaW_hidden_output):
        error = y - finalOutputs
        hiddenError = np.dot(error, self.weights_hidden_output.T)

        outputErrorTerm = error
        hiddenErrorTerm = hiddenError * hiddenOutputs * (1 - hiddenOutputs)

        deltaW_input_hidden     += hiddenErrorTerm * X[:, None]
        deltaW_hidden_output    += outputErrorTerm * hiddenOutputs[:, None]

        return deltaW_input_hidden, deltaW_hidden_output

    def updateWeights(self, deltaW_input_hidden, deltaW_hidden_output, nRecords):
        self.weights_input_hidden   += self.learningRate * deltaW_input_hidden / nRecords
        self.weights_hidden_output  += self.learningRate * deltaW_hidden_output / nRecords


    def run(self, features):
        hiddenInputs = np.dot(features, self.weights_input_hidden)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs     = np.dot(hiddenOutputs, self.weights_hidden_output)
        finalOutputs    = finalInputs

        return finalOutputs

iterations      = 1500
learningRate    = 0.2
hiddenNodes     = 4
outputNodes     = 1 

