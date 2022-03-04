import numpy as np
import scipy
import time


# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # set learning rate
        self.lr = learningrate
        
        # create matrices for the weights wih (input to hidden layer) and who (hidden to output layer)
        # the weights are ordered as w_i_j, which means this weights links node i in the first layer to node j in the next
        # so we get:
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # use the sigmoid function as the activation function for binary classification
        self.activation_function = lambda x: scipy.special.expit(x)

    #train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # error is the (target - actual)
        output_errors = targets - final_outputs
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors* final_outputs *(1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors* hidden_outputs *(1.0 - hidden_outputs)), np.transpose(inputs))


    # make predictions with the neural network
    def predict(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer and calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))
        # calculate signals into final output layer and calculate the signals emerging from final output layer
        final_outputs = self.activation_function(np.dot(self.who, hidden_outputs))
        return final_outputs

#train a given neural network for a given number of epochs
def train_network(nn, epochs, X_train, y_train, X_dev, y_dev, verbose=False):
    #iterate over the number of epochs
    for i in range(epochs):
        #train on each sample in train data
        for n, x in enumerate(X_train):
            start = time.time()
            y = y_train[n]
            nn.train(x,y)
        #if verbose setting is turned on, print update on the training process after each epoch
        if verbose:
            end = time.time()
            print(f"Epoch {i}/{epochs}:")
            print(f"time: {end-start} s; train performance: {test_network(nn, X_train, y_train)[1]}")
            print(f"dev performance: {test_network(nn, X_dev, y_dev)[1]}")

def safe_divide(numerator, denominator):
    #This function divides the numerator by the denominator. If the denominator is zero it returns zero
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0

#tests NN and returns compute performance scores (accuracy, precision, recall, F1-score)
def test_network(nn, X_test, y_test):
    predictions = []
    for x in X_test:
        #correct_label = y_test[n] 
        predicted_label = 0 if nn.predict(x) < 0.5 else 1
        predictions.append(predicted_label)
    scores =  get_performance_scores(y_test, predictions)
    return predictions, scores

#computes and returns accuracy, recall, precicion and F1 score
def get_performance_scores(y_test, predictions):
    #evaluate predictions
    evaluation_counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    for i, annotation in enumerate(y_test):
        evaluation_counts[annotation][predictions[i]] += 1

    #cound true positives, true negatives, false positives and false negatives
    TP = evaluation_counts[1][1]
    TN = evaluation_counts[0][0]
    FP = evaluation_counts[0][1]
    FN = evaluation_counts[1][0]

    #calculate metrics
    precision = round(safe_divide(TP, (TP + FP)),3)
    recall = round(safe_divide(TP, (TP + FN)),3)
    F1 = round(safe_divide((2* precision * recall), (precision + recall)),3)

    accuracy_scores = np.array([1 if prediction == y_test[n] else 0 for n,prediction in enumerate(predictions)])
    accuracy = accuracy_scores.sum() / accuracy_scores.size
    scores = {'accuracy': accuracy,
                'precision' : precision,
                'recall': recall,
                'F1': F1}

    return scores