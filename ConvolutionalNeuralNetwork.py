import numpy as np
from scipy.signal import correlate2d, convolve2d
import time

"Implementation of a Convolutional Neural Network from scratch."

class Convolution():
    "Convolutional layer"

    def __init__(self, input_shape, kernel_size, num_kernels):
        """parameters:
        - input_shape = (input_depth, input_height, input_width): shape of input tensors
        - kernel_size: size of kernel matrices (square matrices)
        - num_kernels: number of kernels (i.e. depth of output tensor)"""

        input_depth, input_height, input_width = input_shape
        self.num_kernels = num_kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (num_kernels, input_height - kernel_size +1, input_width - kernel_size + 1) # shape of ouput tensor/feature maps
        self.kernels_shape = (num_kernels, input_depth, kernel_size, kernel_size)  # shape of kernel tensor
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        "return output tensor from convolutional layer"
        self.input = input_data
        output = np.zeros(self.output_shape)
        for i in range(self.num_kernels): # i indexes number of kernels/depth of output tensor
            for j in range(self.input_depth):   # j indexes depth of input tensor (same as depth of kernels)
                # use scipy to perform cross-correlation
                output[i] += correlate2d(self.input[j], self.kernels[i, j], "valid")
                # output[i] is a (input_height - kernel_size + 1) x (input_width - kernel_size + 1) matrix - known as a feature map
                # input[j] is a input_height x input_width matrix
                # kernels[i, j] is a kernel_size x kernel_size matrix
            output[i] += self.biases[i]

        return output
    
    def backward(self, output_gradient, learning_rate):
        "returns gradient of cost function wrt to input tensor for the layer and updates kernels and biases"
        kernels_gradient = np.zeros(self.kernels_shape)  # gradient of cost wrt kernels
        input_gradient = np.zeros(self.input_shape)   # gradient of cost wrt input tensor

        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                kernels_gradient[i,j] = correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] = convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate*kernels_gradient
        self.biases -= learning_rate*output_gradient

        return input_gradient
    
class MaxPool():
    "Max pooling layer"

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        """Condenses output tensor from convolutional layer. 
        For each feature map (i.e for each kernel), a 'patch' of size (pool_size x pool_size) traces over feature map and 
        outputs the maximum activation within the patch.
        """
        self.input_data = input_data
        self.num_kernels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros((self.num_kernels, self.output_height, self.output_width))

        # loop over all kernels
        for c in range(self.num_kernels):
            # loop over spatial dimensions with stride = pool_size
            for i in range(self.output_height):
                for j in range(self.output_width):

                    start_i = i*self.pool_size  
                    start_j = j*self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    self.output[c, i, j] = np.max(patch)

        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        """Transmit the gradient of the maximum values backward. 
    output_gradient is the gradient of the loss with respect to the output of this layer.
    We propagate the gradient only to the element that was max-pooled during the forward pass.
    """
        input_gradient = np.zeros_like(self.input_data)

        for c in range(self.num_kernels):
            for i in range(self.output_height):
                for j in range(self.output_width):

                    start_i = i*self.pool_size  
                    start_j = j*self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    # Find the max value's indices in the patch
                    max_index = np.unravel_index(np.argmax(patch), patch.shape)

                    # Propagate the gradient from the output to the input at the max location
                    input_gradient[c, start_i + max_index[0], start_j + max_index[1]] = output_gradient[c, i, j]

        return input_gradient
    
class FullyConnected():
    "Fully connected layer"

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)*np.sqrt(1 / self.input_size)
        self.biases = np.random.rand(output_size, 1)

    def softmax(self, z):
        "Softmax activation function"
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)

        return exp_z/np.sum(exp_z, axis=0)
    

    def forward(self, input_data):
        "Returns output from Fully Connected layer"

        self.input_data = input_data    
        flattened_input = input_data.flatten().reshape(1, -1)    # flatten output from previous layer into vector (1, n)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases  # dot product between (10, n) and (n, 1) -> output is (10, 1)
        self.output = self.softmax(self.z)  # Softmax activation - shape (10, 1)

        return self.output
    
    def backward(self, output_gradient, learning_rate):
        "returns gradient of cost function wrt to input for the layer and updates weights and biases"
        dL_dz = output_gradient  # gradient of cost wrt z (pre activation) - same as output gradient when use softmax and cross-entropy loss
                                 # shape (10, 1)

        dL_dw = np.dot(dL_dz, self.input_data.flatten().reshape(1,-1))   # gradient of cost wrt weights
                                                                         # dot product between (10, 1) and (1,n) -> output is (10, n)
        dL_db = dL_dz   # gradient of cost wrt biases

        input_gradient = np.dot(self.weights.T, dL_dz)  # dot product between (n, 10) and (10, 1) -> output is (n, 1)
        input_gradient = input_gradient.reshape(self.input_data.shape)  # reshape to input_data.shape

        self.weights -= dL_dw*learning_rate
        self.biases -= dL_db*learning_rate

        return input_gradient
    
def categorical_cross_entropy_loss(predictions, targets):
    """Calculate the categorical cross-entropy loss between predictions and targets.

    Args:
    predictions (numpy.ndarray): The predicted probabilities for each class (shape: [num_samples, num_classes]).
    targets (numpy.ndarray): The true labels in one-hot encoded format (shape: [num_samples, num_classes]).

    Returns:
    float: The categorical cross-entropy loss.
    """
    num_samples = predictions.shape[0]
    
    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Compute categorical cross-entropy loss
    loss = -np.sum(targets * np.log(predictions)) / num_samples

    return loss

def categorical_cross_entropy_gradient(predictions, targets):
    """Calculate the gradient of the categorical cross-entropy loss. Derivative of loss function wrt softmax has very simple form.

    - predictions (numpy.ndarray): The predicted probabilities for each class (shape: [num_samples, num_classes]).
    - targets (numpy.ndarray): The true labels in one-hot encoded format (shape: [num_samples, num_classes]).
    """
    num_samples = predictions.shape[0]
    gradient = (predictions - targets) / num_samples
    return gradient

def train_network(training_data, conv, pool, full, lr, epochs, mini_batch_size):

    """trains the network using stochastic gradient descent (mini-batches).
    
    - 'training_data': list of tuples (x,y) where x are the input vectors
    and y are the corresponding labels. 
    - 'conv': instance of convolutional layer class
    - 'pool': instance of pooling layer class
    - 'full': instance of fully connected layer class
    - 'lr': learning rate (float)
    - 'epochs' : number of epochs to perform (int)
    - 'mini_batch_size': size of the mini-batches for SGD
    """
    n = len(training_data)  # total number of training examples
    total_time = 0

    for epoch in range(epochs):
        time1 = time.time()
        np.random.shuffle(training_data)  # Shuffle data at the start of each epoch

        # Split data into mini-batches
        mini_batches = [training_data[k:k + mini_batch_size]for k in range(0, n, mini_batch_size)]

        total_loss = 0.0
        correct_predictions = 0

        # Train on each mini-batch
        for mini_batch in mini_batches:
            batch_loss = 0
            batch_correct_predictions = 0

            # Unpack mini-batch into inputs and targets
            x_batch = [x for x, y in mini_batch]
            y_batch = [y for x, y in mini_batch]

            for i in range(len(x_batch)):
                # Forward pass
                conv_out = conv.forward(x_batch[i])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)

                # Loss calculation
                loss = categorical_cross_entropy_loss(full_out, y_batch[i]) # shape (10, 1)
                batch_loss += loss

                # Prediction
                predicted_class = np.argmax(full_out)
                actual_class = np.argmax(y_batch[i])

                if predicted_class == actual_class:
                    batch_correct_predictions += 1

                # Backward pass (update weights and biases)
                gradient = categorical_cross_entropy_gradient(full_out, y_batch[i]) # shape (10, 1)
                full_back = full.backward(gradient, lr)
                pool_back = pool.backward(full_back, lr)
                conv_back = conv.backward(pool_back, lr)

            total_loss += batch_loss
            correct_predictions += batch_correct_predictions

        # After each epoch, calculate average loss and accuracy
        average_loss = total_loss / n
        accuracy = correct_predictions / n
        time2 = time.time()
        t = time2 - time1
        total_time += t

        print(f'Epoch {epoch + 1}/{epochs} completed in {t:.2f} seconds: Loss = {average_loss:.4f}, accuracy = {accuracy*100:.2f}%')
    
    print(f'Training completed in {total_time:.2f} seconds')

def predict(x, conv, pool, full):

    "Returns predicted class for an input x"

    conv_out = conv.forward(x)
    pool_out = pool.forward(conv_out)
    full_out = full.forward(pool_out)
    predicted_class = np.argmax(full_out)

    return predicted_class

def evaluate(test_data, conv, pool, full):

    """Return the fraction of test inputs for which the neural
        network outputs the correct result.
        - 'test_data': list of tuples (x,y) where x are the input vectors
    and y are the corresponding labels. """

    correct_predictions = 0
    for x,y in test_data:
        predicted_class = predict(x, conv, pool, full)
        target_class = np.argmax(y)

        if predicted_class == target_class:
            correct_predictions += 1

    accuracy = correct_predictions/len(test_data)

    return accuracy


















    






                


