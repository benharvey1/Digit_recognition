import numpy as np
import random
import time

"""Implementation of an Artificial Neural Network from scratch. 
Choice of Quadratic or Cross Entropy cost functions. L2 Regularization implemented """

def sigmoid(x):
    "Sigmoid function"
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    "Derivative of sigmoid function"
    return sigmoid(x)*(1 - sigmoid(x))

class QuadraticCost():
    "Class for Quadratic Cost function"

    @staticmethod
    def cost(a, y):
        "Return the cost asscoiated with ouput 'a' and desired output 'y'."
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def derivative(z, a, y):
        "Return the derivative dC/da where 'C' is the quadratic cost for a single input and 'a' is the activation of the final layer "
        return a-y
    
class CrossEntropyCost():
    """Class for Cross Entropy Cost function. Cross entropy cost function avoids 'learning slowdown' which occurs with quadratic cost
    Since 'sigmoid_prime' term in denominator of derivative cancels out with 'sigmoid_prime' term from derivative da/dz."""

    @staticmethod
    def cost(a, y):
        "Return the cost asscoiated with ouput 'a' and desired output 'y'."
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    @staticmethod
    def derivative(z, a, y):
        "Return the derivative dC/da where 'C' is the quadratic cost for a single input and 'a' is the activation of the final layer "
        return (a - y)/sigmoid_prime(z)

class NeuralNetwork():

    def __init__(self, sizes, cost=CrossEntropyCost):
        """ - List 'sizes' contains number of neurons in network. If list was [2, 4, 2] then it would be 3-layer network 
        with 2 neurons in first layer, 4 in second layer and 3 in final layer. The first layer of the network must have 
        the same number of neurons as dimension of the input data.
            - Biases initialised using a normal distribution with mean 0 and standard deviation 1
            - Weights initialised using normal distribution with mean 0 and standard deviation 1/sqrt(n_int) wher n_int is the number of
            input neurons to that layer
        """
        self.sizes = sizes
        self.cost = cost
        self.num_layers = len(sizes)
        self.biases = [np.random.normal(loc=0.0, scale=1, size=(i,1)) for i in sizes[1:]]
        self.weights = [np.random.normal(loc=0.0, scale=1/np.sqrt(i), size=(j,i)) for i,j in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        "Return output of network if 'a' is input vector"
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)    # a is (n, 1) Numpy ndarray not a (n,) vector
        return a
    
    def backpropagation(self, x, y):
        "return a tuple (nabla_b, nabla_w) representing the gradient of the C with respect to the biases and weights of each layer."
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        a = x   # actiavtion for input layer
        activations = [x]   # stores activations of each layer. For first layer activation is just input vector
        z_vectors = []  # stores vectors z = w.a + b for each layer

        # forward pass
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b    # a (and hence input vectors and corresponding labels) must be (n,1) numpy.ndarrays not (n,) vectors
            a = sigmoid(z)
            z_vectors.append(z)
            activations.append(a)

        # Compute derivatives for final layer l=L
        delta = self.cost.derivative(z_vectors[-1], activations[-1], y)*sigmoid_prime(z_vectors[-1])   # delta^(L)_i = (dC/da^(L)_i)*(sigmoid'(z^(L)_i)) 
        nabla_b[-1] = delta # dC/db_i = delta_i
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())    # dC/dw^(L)_ij = (delta^(L)_i)(a^(L-1)_j)

        # use back propagation to compute nabla_b and nabla_w at each layer in the network
        for i in range(2, self.num_layers):
            z = z_vectors[-i]
            delta = sigmoid_prime(z)*np.dot(self.weights[-i+1].transpose(), delta)  # delta^(l)_i = sigmoid'(z^(l)_i)*[((w^(l+1))^T delta^(l+1))_i]
            nabla_b[-i] = delta
      
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def update_mini_batch(self, mini_batch, learning_rate, reg_param, n):
        """Update weights and biases in the 'mini batch' (subset of training data) using gradient descent and back propagation.
        - 'mini_batch' is a list of tuples (x,y) where x is input vector and y is corresponding label
        - reg_param is L2 regularization parameter. This helps reduce overfitting."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)   # calculate gradient of C (wrt weights and biases) at point (x,y) using back propagation
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # Use fact that cost function is equal to average of C over all input vectors
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]   # Where C is cost function for a single input vector

        self.weights = [w*(1-learning_rate*(reg_param/n)) - learning_rate/len(mini_batch)*nw for w, nw in zip(self.weights, nabla_w)]   # update weights and biases using gradient descent
        self.biases = [b - learning_rate/len(mini_batch)*nb for b, nb in zip(self.biases, nabla_b)]
    

    def train(self, training_data, epochs, learning_rate, mini_batch_size, reg_param):
        """Train the network using stochastic gradient descent. 
            -'training data' is a list of tuples (x,y) where x are the input vectors
         and y are the corresponding labels (one hot representation). 
            - 'mini_batch_size' is the size of the subsets which training data is split into
            - 'epochs' is the number of complete iterations   """

        n = len(training_data)
        total_time = 0
        for i in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)   # randomize order of elements in list 'training_data'
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0,n,mini_batch_size)] # split training data into mini batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, reg_param, n) # update biases and weights in each mini batch using gradient descent
            time2 = time.time()
            t = time2 - time1
            total_time += t

            print(f"Epoch {i+1}/{epochs} completed in {t:.2f} seconds")

        print(f"Training completed in {total_time:.2f} seconds")
    
    def evaluate(self, test_data):
        """Return the fraction of test inputs for which the neural
        network outputs the correct result. Network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        correct = 0
        for (x, y) in results:
            if x == y:
                correct += 1
        
        return correct/len(results)






