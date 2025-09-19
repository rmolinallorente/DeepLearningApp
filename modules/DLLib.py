
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import math
import sklearn
import sklearn.datasets


 
def load_extra_datasets():
    N = 300
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
 

def load_datasetSK():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=200, noise=.2) #300 #0.2 
    
    return train_X, train_Y




def load_dataset(file_pathD,file_pathT):
    train_dataset = h5py.File(file_pathT, "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(file_pathD, "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes


# GRADED FUNCTION: softmax

def softmax(Z):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    #Z_exp = np.exp(Z)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    #Z_sum = np.sum(Z_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    #A = Z_exp / Z_sum
    
    A = Z#np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims = True)
    cache = Z

    ### END CODE HERE ###
    
    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

# GRADED FUNCTION: sigmoid

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache
    
    
def tanh(Z):
    """
    Implements the tanh activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of tanh(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    #print("Z:", Z)    
    #A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    
    A = 2/(1 + np.exp(-2*Z)) - 1
    
   
    #print("AZZ:", A)
    cache = Z

    return A, cache    
    
    
def linear(Z):
    """
    Implements the linear activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of linear(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = Z
    cache = Z

    return A, cache        

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ
    
def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    dZ = dA * (1-s**2)

    assert (dZ.shape == Z.shape)

    return dZ  

def linearFunc_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    #dZ = dA 

    assert (dZ.shape == Z.shape)

    return dZ 
    
def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a single SOFTMAX unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    #dZ=dA
    
    #s = 1/(1+np.exp(-Z))
    #dZ = dA * s * (1-s)
    """print(len(dA))
    for i in range(len(dA)):
        for j in range(len(dA)):
            if i == j:
                dZ[i] = dA[i] * (1-dA[i])
            else: 
                 dZ[i] = -dA[i]*dA[j]"""
    
    #s = 1/(1+np.exp(-Z))
    #dZ = dA * s * (1-s)
    
    
    exps = np.exp(Z)
    a = exps / np.sum(exps, axis = 0, keepdims=True)
 
    dA_a = np.sum(dA * a, axis = 0)
 
    dZ = a * (dA - dA_a)
    
   
    assert (dZ.shape == Z.shape)

    return dZ    

# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    #(≈ 4 lines of code)
    # W1 = ...
    # b1 = ...
    # W2 = ...
    # b2 = ...
    # YOUR CODE STARTS HERE
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    # YOUR CODE ENDS HERE

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# GRADED FUNCTION: initialize_parameters_deep

def initialize_parameters_deep_zeros(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims) # number of layers in the network
    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = ...
        # parameters['b' + str(l)] = ...
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        # YOUR CODE ENDS HERE

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters

def initialize_parameters_deep_random(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = ...
        # parameters['b' + str(l)] = ...
        # YOUR CODE STARTS HERE
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        # YOUR CODE ENDS HERE

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters
    
def initialize_parameters_deep_he(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = ...
        # parameters['b' + str(l)] = ...
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))        
        # YOUR CODE ENDS HERE

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters    


def compute_costMiniBatch(AL, Y, cost_function):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    # (≈ 1 lines of code)
    # cost = ...
    # YOUR CODE STARTS HERE
    #cost=-1/m*np.sum(np.dot(Y,np.log(AL))+(1-Y)*np.log(1-A))
    if cost_function == "mse":
        cost = np.sum(np.multiply((AL - Y),(AL - Y)))
    else:
        cost = (-1 / 1) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    


    # YOUR CODE ENDS HERE

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).


    return cost

def compute_total_lossMiniBatch(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(Y), tf.transpose(AL),from_logits=True))
   
     # computing softmax values for predicted values
    """AL = np.exp(AL) / np.sum(np.exp(AL))##, axis=1, keepdims = True)
    cost = 0    
    # Doing cross entropy Loss
    for i in range(len(AL)):
 
        # Here, the loss is computed using the
        # above mathematical formulation.
        cost = cost + (-1 * Y[i]*np.log(AL[i]))
        
    print(total_loss)""" 
    

    # YOUR CODE ENDS HERE

    total_loss = np.squeeze(total_loss)      # To make sure your total_loss's shape is what we expect (e.g. this turns [[17]] into 17).


    return total_loss


def compute_total_loss(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    
    
    #labels = tf.constant([[1.0, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    #logits = tf.constant([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    
    y_pred = tf.constant([0.1, 0.3, 0.4, 0.2])
    y_hat =tf.constant([0, 1, 0, 0])
    
   
    
    labels = Y
    logits = AL

    
    #cross_entropy = - np.sum(np.log(y_pred)*y_hat)
    
    sum_score = 0.0
    for i in range(len(logits)):
        for j in range(len(logits[i])):
            sum_score += logits[i][j] * np.log(1e-15 + labels[i][j])
    mean_sum_score = 1.0 / len(logits) * sum_score            
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy( tf.transpose(labels), tf.transpose(logits), from_logits=True))
    #print(total_loss)
    #print(len(logits))
    #print(-mean_sum_score)
    total_loss = total_loss/m
     # computing softmax values for predicted values
    """AL = np.exp(AL) / np.sum(np.exp(AL))##, axis=1, keepdims = True)
    cost = 0    
    # Doing cross entropy Loss
    for i in range(len(AL)):
 
        # Here, the loss is computed using the
        # above mathematical formulation.
        cost = cost + (-1 * Y[i]*np.log(AL[i]))
        
    print(total_loss)""" 
    

    total_loss = np.squeeze(total_loss)      # To make sure your total_loss's shape is what we expect (e.g. this turns [[17]] into 17).


    return total_loss

def compute_cost(AL, Y, cost_function):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    # Compute loss from aL and y.
    # (≈ 1 lines of code)
    # cost = ...
    if cost_function == "mse":
        cost = (1. / m) * np.sum(np.multiply((AL - Y),(AL - Y)))
    else:    
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        #logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        #cost = 1./m * np.nansum(logprobs)
  
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost



def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    #(≈ 1 line of code)
    # Z = ...          
    Z = np.dot(W,A)+b
    cache = (A, W, b)

    return Z, cache

# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
        # YOUR CODE ENDS HERE
        
    if activation == "tanh":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = tanh(Z)
        # YOUR CODE ENDS HERE        

    elif activation == "relu":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    elif activation == "softmax":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = softmax(Z)

    elif activation == "linear":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = linear(Z)


        # YOUR CODE ENDS HERE
    cache = (linear_cache, activation_cache)

    return A, cache

# GRADED FUNCTION: compute_cost

def L_model_forward(X, keep_prob_list, parameters, activation, activationL):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    np.random.seed(1)
    caches = []
    Dx = {}
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    #print(type(keep_prob))   
    #print(type(keep_prob))
    #print(keep_prob_list)
    for l in range(1, L):
        A_prev = A
        #(≈ 2 lines of code)
        # A, cache = ...
        # caches ...             
        if l >= 2:     
            #print(keep_prob[l-2])
            if keep_prob_list[l-2] < 1.0:
                Dx["D" + str(l-1)] = np.random.rand(A_prev.shape[0],A_prev.shape[1])
                Dx["D" + str(l-1)] = Dx["D" + str(l-1)] < keep_prob_list[l-2]
                A_prev = np.multiply(A_prev,Dx["D" + str(l-1)])        
                A_prev = A_prev / keep_prob_list[l-2]           
        
        if activation == "relu":        
            A, cache = linear_activation_forward(A_prev, (parameters["W"+ str(l)]), (parameters["b"+ str(l)]), activation = "relu")
        else:
            A, cache = linear_activation_forward(A_prev, (parameters["W"+ str(l)]), (parameters["b"+ str(l)]), activation = "tanh")
        
        #if l >= 2:          
        if keep_prob_list[l-1] < 1.0:
            cache = (cache , Dx) 
        
        caches.append(cache)
        #print(Dx[D2])
        #Dx[l] = D_temp
     

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    #(≈ 2 lines of code)
    # AL, cache = ...
    # caches ...
    if keep_prob_list[L-2] < 1.0:
        Dx["D" + str(L-1)] = np.random.rand(A.shape[0],A.shape[1])   
        Dx["D" + str(L-1)] = Dx["D" + str(L-1)] < keep_prob_list[L-2]    
        A = np.multiply(A,Dx["D" + str(L-1)])    
        A = A / keep_prob_list[L-2]
    

    if activationL == "sigmoid": 
        AL, cache = linear_activation_forward(A, (parameters["W"+ str(L)]), (parameters["b"+ str(L)]), activation = "sigmoid")
    else:
        AL, cache = linear_activation_forward(A, (parameters["W"+ str(L)]), (parameters["b"+ str(L)]), activation = "linear")
      
    if keep_prob_list[L-2] < 1.0:
        cache = (cache , Dx)
    caches.append(cache)

    return AL, caches



def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache    
    m = A_prev.shape[1]
       
    dW = 1./m * np.dot(dZ,cache[0].T) + (lambd/m)*W
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache[1].T,dZ)

    return dA_prev, dW, db
    
    

# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, lambd, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
   
    linear_cache, activation_cache = cache

    if activation == "relu":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache, lambd)

    elif activation == "sigmoid":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "tanh":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache, lambd)
    
    elif activation == "softmax":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "linear":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        dZ = linearFunc_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches, cost_function, lambd, keep_prob_list, activation, activationL):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    #(1 line of code)
    # dAL = ...

    if cost_function == "mse":
        dAL = 2*(AL - Y)
    else:
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        #dAL = AL - Y


    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    # current_cache = ...
    # dA_prev_temp, dW_temp, db_temp = ...
    # grads["dA" + str(L-1)] = ...
    # grads["dW" + str(L)] = ...
    # grads["db" + str(L)] = ...  
    if keep_prob_list[L-2] < 1.0:
        current_cache, D = caches[L-1]
    else:
        current_cache = caches[L-1]
    if activationL == "sigmoid":    
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, lambd, activation = "sigmoid")
        #dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation = "softmax")
    else:
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, lambd, activation = "linear")


    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp


    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        # current_cache = ...
        # dA_prev_temp, dW_temp, db_temp = ...
        # grads["dA" + str(l)] = ...
        # grads["dW" + str(l + 1)] = ...
        # grads["db" + str(l + 1)] = ...
                    
        if keep_prob_list[l] < 1.0:
            current_cache, D = caches[l]
            D = D["D" + str(l + 1)] 
            grads["dA" + str(l + 1)] = np.multiply(grads["dA" + str(l + 1)],D)
            grads["dA" + str(l + 1)] = grads["dA" + str(l + 1)] / keep_prob_list[l]            
        else:
            current_cache = caches[l]  
            
        if activation == "relu":
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, lambd, activation = "relu")
        else:
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, lambd, activation = "tanh")
                    
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp


    return grads



# GRADED FUNCTION: random_mini_batches

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]    
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    
    inc = mini_batch_size

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        # (approx. 2 lines)
        # mini_batch_X =  
        # mini_batch_Y =
        # YOUR CODE STARTS HERE
        mini_batch_X = shuffled_X[:,mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*k:mini_batch_size*(k+1)]
        
        
        # YOUR CODE ENDS HERE
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        #(approx. 2 lines)
        # mini_batch_X =
        # mini_batch_Y =
        # YOUR CODE STARTS HERE
        mini_batch_X = shuffled_X[ :, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[ :, num_complete_minibatches*mini_batch_size : ]
        
        # YOUR CODE ENDS HERE
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    #(≈ 2 lines of code)
    for l in range(L):
        # parameters["W" + str(l+1)] = ...
        # parameters["b" + str(l+1)] = ...
        # YOUR CODE STARTS HERE
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        # YOUR CODE ENDS HERE
    return parameters

# GRADED FUNCTION: two_layer_model


def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict_dec(parameters, X, activation, activationL):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    keep_prob_temp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    a3, cache = L_model_forward(X, keep_prob_temp, parameters, activation, activationL)
    predictions = (a3 > 0.5)
    return predictions

def predict2(X, parameters, activation, activationL):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    keep_prob_temp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    probas, caches = L_model_forward(X, keep_prob_temp, parameters, activation, activationL)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

  
    return p

def predict3(X, parameters, activation, activationL):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    # Forward propagation
    keep_prob_temp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    probas, caches = L_model_forward(X, keep_prob_temp, parameters, activation, activationL)
    #print(X)
    #print(parameters)

     
    return probas


def predict(X, y, parameters, activation, activationL):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    keep_prob_temp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    probas, caches = L_model_forward(X, keep_prob_temp, parameters, activation, activationL)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    Acc = np.sum((p == y)/m)
    #print("Accuracy: "  + str(Acc))

    return p, Acc
    
def predictClass(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)


    return probas    

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    
    Returns:
    cost -- the cost function (logistic cost for m examples)
    cache -- a tuple with the intermediate values (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    """
    
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(log_probs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:245760].reshape((20,12288))
    parameters["b1"] = theta[245760:245780].reshape((20,1))
    parameters["W2"] = theta[245780:245920].reshape((7,20))
    parameters["b2"] = theta[245920:245927].reshape((7,1))
    parameters["W3"] = theta[245927:245962].reshape((5,7))
    parameters["b3"] = theta[245962:245967].reshape((5,1))
    parameters["W4"] = theta[245967:245972].reshape((1,5))
    parameters["b4"] = theta[245972:245973].reshape((1,1))    

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3", "dW4", "db4"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters 
    X -- input datapoint, of shape (input size, number of examples)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    #for i in range(num_parameters):
    for i in range(1000):
        
        print(i)
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have outputs two parameters but we only care about the first one
        #(approx. 3 lines)
        # theta_plus =                                        # Step 1
        # theta_plus[i] =                                     # Step 2
        # J_plus[i], _ =                                     # Step 3
        # YOUR CODE STARTS HERE
        theta_plus = np.copy(parameters_values)
        theta_plus[i] =  theta_plus[i] + epsilon        
        AL, _ = L_model_forward(X, vector_to_dictionary(theta_plus))        
        J_plus[i] = compute_cost(AL, Y, cost_function = "binaryCross")
         
        
        # YOUR CODE ENDS HERE
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        #(approx. 3 lines)
        # theta_minus =                                    # Step 1
        # theta_minus[i] =                                 # Step 2        
        # J_minus[i], _ =                                 # Step 3
        # YOUR CODE STARTS HERE
        theta_minus = np.copy(parameters_values)
        theta_minus[i] =  theta_minus[i] - epsilon
        #_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))
        AL, _ = L_model_forward(X, vector_to_dictionary(theta_minus))        
        J_minus[i] = compute_cost(AL, Y, cost_function = "binaryCross")
        
        # YOUR CODE ENDS HERE
        
        # Compute gradapprox[i]
        # (approx. 1 line)
        # gradapprox[i] = 
        # YOUR CODE STARTS HERE
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)
        
        # YOUR CODE ENDS HERE
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    # (approx. 3 line)
    # numerator =                                             # Step 1'
    # denominator =                                           # Step 2'
    # difference =                                            # Step 3'
    # YOUR CODE STARTS HERE
    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    # YOUR CODE ENDS HERE
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


# GRADED FUNCTION: compute_cost_with_regularization


def compute_cost_with_regularizationMiniBatch(AL, Y, parameters, lambd, cost_function):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]

    """parameters["W1"]=[[3, 4], [3, 3]]
    parameters["W2"]=[[3, 4], [3, 3]]
    parameters["W3"]=[[3, 4], [3, 3]]
    parameters["W4"]=[[3, 4], [3, 3]]"""
    """W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]"""
    
    cross_entropy_cost = compute_costMiniBatch(AL, Y, cost_function) # This gives you the cross-entropy part of the cost
    
    
    
    L = len(parameters) // 2                 # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    L2_regularization_cost = 0    
    for l in range(1, L+1):                
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square((parameters["W"+ str(l)])))
            

    #L2_regularization_cost = (L2_regularization_cost * lambd)/(2.0*m)
    L2_regularization_cost = (L2_regularization_cost * lambd)/(2.0)
    #L2_regularization_cost2 = lambd/(2.0*m)*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3))+np.sum(np.square(W4)))    
        
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd, cost_function):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]

    """parameters["W1"]=[[3, 4], [3, 3]]
    parameters["W2"]=[[3, 4], [3, 3]]
    parameters["W3"]=[[3, 4], [3, 3]]
    parameters["W4"]=[[3, 4], [3, 3]]"""
    """W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]"""
    
    cross_entropy_cost = compute_cost(AL, Y, cost_function) # This gives you the cross-entropy part of the cost
    
    
    
    L = len(parameters) // 2                 # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    L2_regularization_cost = 0    
    for l in range(1, L+1):                
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square((parameters["W"+ str(l)])))
            

    L2_regularization_cost = (L2_regularization_cost * lambd)/(2.0*m)
    #L2_regularization_cost2 = lambd/(2.0*m)*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3))+np.sum(np.square(W4)))    
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost



# GRADED FUNCTION: schedule_lr_decay

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    # (approx. 1 lines)
    # learning_rate = ...
    # YOUR CODE STARTS HERE
    learning_rate = learning_rate0/(1+decay_rate*np.floor(epoch_num/time_interval))
    
    # YOUR CODE ENDS HERE
    return learning_rate


# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(1, L + 1):
        # (approx. 2 lines)
        # v["dW" + str(l)] =
        # v["db" + str(l)] =
        v["dW" + str(l)] = np.zeros( parameters['W'+str(l)].shape )
        v["db" + str(l)] = np.zeros( parameters['b'+str(l)].shape )
        
        
    return v

# GRADED FUNCTION: update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(1, L + 1):
        
        # (approx. 4 lines)
        # compute velocities
        # v["dW" + str(l)] = ...
        # v["db" + str(l)] = ...
        # update parameters
        # parameters["W" + str(l)] = ...
        # parameters["b" + str(l)] = ...
        v["dW" + str(l)] = np.multiply(beta,v["dW" + str(l)]) + np.multiply((1-beta),grads["dW" + str(l)])
        v["db" + str(l)] = beta * v["db" + str(l)] + (1-beta)*grads["db" + str(l)]
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate *  v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate *  v["db" + str(l)]
        
    return parameters, v

# GRADED FUNCTION: initialize_adam

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):
    # (approx. 4 lines)
        # v["dW" + str(l)] = ...
        # v["db" + str(l)] = ...
        # s["dW" + str(l)] = ...
        # s["db" + str(l)] = ...
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    return v, s


# GRADED FUNCTION: update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- Adam variable, counts the number of taken steps
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        # (approx. 2 lines)
        # v["dW" + str(l)] = ...
        # v["db" + str(l)] = ...
        v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*grads['dW' + str(l)]
        v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*grads['db' + str(l)]
        
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        # (approx. 2 lines)
        # v_corrected["dW" + str(l)] = ...
        # v_corrected["db" + str(l)] = ...
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1-np.power(beta1,t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1-np.power(beta1,t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        #(approx. 2 lines)
        # s["dW" + str(l)] = ...
        # s["db" + str(l)] = ...
        s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1-beta2) * np.power(grads['dW' + str(l)],2)
        s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2) * np.power(grads['db' + str(l)],2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        # (approx. 2 lines)
        # ...
        # s_corrected["db" + str(l)] = ...
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1-np.power(beta2,t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1-np.power(beta2,t))
        
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        # (approx. 2 lines)
        # parameters["W" + str(l)] = ...
        # parameters["b" + str(l)] = ...
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)


    return parameters, v, s, v_corrected, s_corrected


# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, keep_prob_list, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, optimizer = "momentum", learning_rate = 0.0075, 
                    num_epochs = 5000,  print_cost=False, initialization = "he", lambd = 0, decay=False, time_interval = 1000, 
                    activation = "relu", activationL = "sigmoid", cost_function = "binaryCross"):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    learning_rate0 = learning_rate   # the original learning rate
    np.random.seed(1)
    costs = []                         # keep track of cost
    t = 0                            # initializing the counter required for Adam update

    # Parameters initialization.
    #(≈ 1 line of code)
    # parameters = ...
     
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_deep_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_deep_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_deep_he(layers_dims)
        
        # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)        
   
    
    # Optimization loop
    for i in range(num_epochs):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        # AL, caches = ...       
        AL, caches = L_model_forward(X, keep_prob_list, parameters, activation, activationL)
          
        if lambd == 0:
            cost = compute_cost(AL, Y, cost_function)  
            #cost = compute_total_loss(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd, cost_function)        
        # Backward propagation.
        # grads = ...
        
        grads = L_model_backward(AL, Y, caches, cost_function, lambd, keep_prob_list, activation, activationL)       
    
        # Update parameters.
        #(≈ 1 line of code)
        # parameters = ...
        # Update parameters
        #print(grads)
        if optimizer == "gd":
            parameters = update_parameters(parameters, grads, learning_rate)
        elif optimizer == "momentum":
            parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
        elif optimizer == "adam":
            t = t + 1 # Adam counter
            parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                           t, learning_rate, beta1, beta2,  epsilon)   
        
        if decay:
            epoch_num = i
            decay_rate = 0.3
            learning_rate=schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval)
        
        #if i<2:
         #   difference = gradient_check_n(parameters, grads, X, Y, 1e-7, True)

        if num_epochs > 500:
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)                     
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_epochs:
                costs.append(cost)
        elif num_epochs > 100:
            # Print the cost every 10 iterations
            if print_cost and i % 10 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)      
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)))      
            if i % 10 == 0 or i == num_epochs:
                costs.append(cost)
        else:
            # Print the cost every 1 iterations
            if print_cost and i % 1 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)      
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost)))       
            if i % 1 == 0 or i == num_epochs:
                costs.append(cost)     

    return parameters, costs

# GRADED FUNCTION: L_layer_model

def L_layer_modelMiniBatch(X, Y, layers_dims, keep_prob_list, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, optimizer = "momentum", learning_rate = 0.0075, 
                            num_epochs = 5000, mini_batch_size = 64, print_cost=False, initialization = "he", lambd = 0, decay=False, time_interval = 1000,
                            activation = "relu", activationL = "sigmoid", cost_function = "binaryCross"):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    learning_rate0 = learning_rate   # the original learning rate
    np.random.seed(1)
    costs = []                         # keep track of cost
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    t = 0                            # initializing the counter required for Adam update
    m = X.shape[1]                   # number of training examples

    # Parameters initialization.
    #(≈ 1 line of code)
    # parameters = ...
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_deep_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_deep_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_deep_he(layers_dims)
    
    
        # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)       
        """print("shape of the 1st mini_batch_X: " + str(minibatches[0][0].shape))
        print("shape of the 2nd mini_batch_X: " + str(minibatches[1][0].shape))
        print("shape of the 3rd mini_batch_X: " + str(minibatches[2][0].shape))
        print("shape of the 1st mini_batch_Y: " + str(minibatches[0][1].shape))
        print("shape of the 2nd mini_batch_Y: " + str(minibatches[1][1].shape))
        print("shape of the 3rd mini_batch_Y: " + str(minibatches[2][1].shape))
        print("mini batch sanity check: " + str(minibatches[0][0][0][0:3]))"""
        
        cost = 0
        
        #print(f"\rEpoch {i}", end="")
        for minibatch in minibatches:

            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation            
            AL, caches = L_model_forward(minibatch_X, keep_prob_list, parameters, activation, activationL)
            #print(parameters)
            #print("AL:",AL)
            #print("minibatch_X:",minibatch_X)

            # Compute cost and add to the cost total
            if lambd == 0:
                cost += compute_costMiniBatch(AL, minibatch_Y, cost_function)                    
                #print("AL:",AL)
                #print("Y:", minibatch_Y)
                #cost += compute_total_lossMiniBatch(AL, minibatch_Y)
            else:
                cost += compute_cost_with_regularizationMiniBatch(AL, minibatch_Y, parameters, lambd, cost_function)            

            # Backward propagation
            grads = L_model_backward(AL, minibatch_Y, caches, cost_function, lambd, keep_prob_list, activation, activationL)
            
            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)            
        cost_avg = cost / m
        if decay:
            epoch_num = i
            decay_rate = 0.3
            learning_rate=schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval)
        
        #if i<2:
         #   difference = gradient_check_n(parameters, grads, X, Y, 1e-7, True)
            

        if num_epochs > 500:
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)      
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg)))     
            if i % 100 == 0 or i == num_epochs:
                costs.append(cost_avg)
        elif num_epochs > 100:
            # Print the cost every 10 iterations
            if print_cost and i % 10 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)      
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg))) 
            if i % 10 == 0 or i == num_epochs:
                costs.append(cost_avg)
        else:
            # Print the cost every 1 iterations
            if print_cost and i % 1 == 0 or i == num_epochs - 1:
                if cost_function == "binaryCross":
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg)), "learning rate: %f"%(learning_rate), end=" ")                
                    pred_train, AccTrain = predict(X, Y, parameters, activation, activationL)      
                    print("Accuracy Train {:.6f}".format(np.squeeze(AccTrain)))
                else:
                    print("Cost after epoch {}: {:.8f}".format(i, np.squeeze(cost_avg)))         
            if i % 1 == 0 or i == num_epochs:
                costs.append(cost_avg)     
      
                

    return parameters, costs



def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

