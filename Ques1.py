import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 3  # number of hidden units in each of the two hidden layers
epochs = 900
learnrate = 0.005
np.random.seed(21)

#Randomly selected input
features = np.random.random_integers(30, size=(50,2))
targets = np.random.random_integers(10, size=(50,1)) /10.
n_records, n_features = features.shape
last_loss = None

# Initialize weights
W1 = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden)) 
W2 = np.random.normal(scale=1 / n_features ** .5, size=(n_hidden, n_hidden))
W3 = np.random.normal(scale=1 / n_features ** .5, size=(n_hidden,1))
                                         

# Initialize bias                                         
b1=np.zeros(shape=(1,n_hidden));
b2=np.zeros(shape=(1,n_hidden));
b3=np.zeros(shape=(1,1));

for e in range(epochs):
    del_w_1 = np.zeros(W1.shape)
    del_w_2 = np.zeros(W2.shape)
    del_w_3 = np.zeros(W3.shape)
    del_b_1 = np.zeros(b1.shape)
    del_b_2 = np.zeros(b2.shape)
    del_b_3 = np.zeros(b3.shape)
    for x, y in zip(features, targets):
        ## Forward pass ##
        # Calculate the output
        x=np.reshape(x, (1,2))
        y=np.reshape(y, (1,1))
        hidden_1_input = np.add(np.dot(x, W1), b1)
        hidden_1_output = sigmoid(hidden_1_input)
        
        hidden_2_input = np.add(np.dot(hidden_1_output, W2), b2)
        hidden_2_output = sigmoid(hidden_2_input)
        
        last_input = np.add(np.dot(hidden_2_output, W3), b3)
        output = sigmoid(last_input)

        ## Backward pass ##
        # Calculate the network's prediction error
        error = y-output

        # Calculate error term for the output unit
        #The derivative of y=sigmoid(x) is y*(1-y)
        output_error_term = error*(output*(1-output))

        ## propagate errors to second hidden layer

        # Calculate the second hidden layer's contribution to the error
        hidden_2_error = output_error_term.dot(W3.T)
        
        # Calculate the error term for the second hidden layer
        hidden_2_error_term = hidden_2_error*(hidden_2_output*(1-hidden_2_output))
        
        # Calculate the first hidden layer's contribution to the error
        hidden_1_error = hidden_2_error_term.dot(W2.T)
        
        # Calculate the error term for the first hidden layer
        hidden_1_error_term = hidden_1_error*(hidden_1_output*(1-hidden_1_output))
        
        # Update the change in weights
        del_w_3 += hidden_2_output.T.dot(output_error_term)
        del_w_2 += hidden_1_output.T.dot(hidden_2_error_term)
        del_w_1 += x.T.dot(hidden_1_error_term)
        del_b_3 += output_error_term
        del_b_2 += hidden_2_error_term
        del_b_1 += hidden_1_error_term

    # Update weights
    W1 += learnrate*del_w_1/n_records
    W2 += learnrate*del_w_2/n_records
    W3 += learnrate*del_w_3/n_records
    b1 += learnrate*del_b_1/n_records
    b2 += learnrate*del_b_2/n_records
    b3 += learnrate*del_b_3/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_1_input = np.add(np.dot(x, W1), b1)
        hidden_1_output = sigmoid(hidden_1_input)
        
        hidden_2_input = np.add(np.dot(hidden_1_output, W2), b2)
        hidden_2_output = sigmoid(hidden_2_input)
        
        last_input = np.add(np.dot(hidden_2_output, W3), b3)
        out = sigmoid(last_input)
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss
