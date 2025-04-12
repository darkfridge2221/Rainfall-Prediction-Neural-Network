import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the preprocessed dataset
data = pd.read_excel("C:/Users/User/Downloads/preprocessed_data.xlsx", sheet_name=None)

#Extract training, validation, and test sets
train = data['Train']
val = data['Validation']
test = data['Test']

#User input for predictors, predictand, and hidden neurons
print("Available columns:",train.columns)  
predictors = input("Enter predictor columns (comma-separated): ").split(',')
predictors = [col.strip() for col in predictors]  #Strip spaces from user input
predictand = input("Enter predictand column: ").strip()
n_hidden = int(input("Enter number of hidden neurons: "))

#Prepare data and reshape target to a column vector
X_train, y_train = train[predictors].values, train[predictand].values.reshape(-1, 1)
X_val, y_val = val[predictors].values, val[predictand].values.reshape(-1, 1)
X_test, y_test = test[predictors].values, test[predictand].values.reshape(-1, 1)

#Initialise network parameters
np.random.seed(42)
n_input, n_output = 3, 1 #Number of neurons
W1 = np.random.uniform(-1, 1, (n_input, n_hidden)) 
W2 = np.random.uniform(-1, 1, (n_hidden, n_output)) #Weights
B1 = np.zeros((1, n_hidden))
B2 = np.zeros((1, n_output)) #Biases

#Initialise momentum terms initially to 0
prev_dW1 = np.zeros_like(W1)
prev_dW2 = np.zeros_like(W2)
prev_dB1 = np.zeros_like(B1)
prev_dB2 = np.zeros_like(B2)

#Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#Training parameters
epochs = 20000
learning_rate = 0.02
momentum = 0.4
min_lr, max_lr = 0.01, 0.1
weight_decay = 0.0001
batch_size = 20

#Store previous error for bold driver
prev_error = float("inf")

#Training loop
for epoch in range(epochs):
    #Shuffle training data
    indices = np.random.permutation(X_train.shape[0]) #Generate a random permutation of indices
    X_train_shuffled = X_train[indices] #Shuffle the training input data
    y_train_shuffled = y_train[indices] #Shuffle the training target data

    for i in range(0, X_train.shape[0], batch_size): #Iterate through training data in batches
        X_batch = X_train_shuffled[i:i+batch_size] #Extract batch of input data from shuffled data
        y_batch = y_train_shuffled[i:i+batch_size] #Extract batch of target data from shuffled data

        #Forward pass
        Z1 = np.dot(X_train, W1) + B1 #Calulate weighted sum of inputs at hidden layer
        A1 = sigmoid(Z1) #Apply sigmoid function to hidden layer
        Z2 = np.dot(A1, W2) + B2 #Calculate weighted sum of inputs at output layer
        A2 = sigmoid(Z2) #Apply sigmoid function to output layer
        
        #Compute error
        error = y_train - A2 #Difference between actual and predicted values
        loss = np.mean(error**2) #Calculate mean squared error
        rmse = np.sqrt(loss) #Calculate root mean squared error
        
        #Backpropagation
        dA2 = error * sigmoid_derivative(A2) #Calculate derivative of activation function at output layer
        dW2 = np.dot(A1.T, dA2) #Calculate gradient of weights between input and output layer
        dB2 = np.sum(dA2, axis=0, keepdims=True) #Calculate gradient of bias at output layer
        dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1) #Calculate derivative of activation function at hidden layer
        dW1 = np.dot(X_train.T, dA1) #Calculate gradient of weights between input and hidden layer
        dB1 = np.sum(dA1, axis=0, keepdims=True) #Calculate gradient of bias at hidden layer

        #Apply weight decay to W1 and W2
        dW1 = dW1 - weight_decay * W1 
        dW2 = dW2 - weight_decay * W2 

        #Apply momentum to weight updates
        dW1 = momentum * prev_dW1 + (1 - momentum) * dW1
        dW2 = momentum * prev_dW2 + (1 - momentum) * dW2
        dB1 = momentum * prev_dB1 + (1 - momentum) * dB1
        dB2 = momentum * prev_dB2 + (1 - momentum) * dB2
        
        #Update weights
        W1 += learning_rate * dW1 #Update weights between input and hidden layer
        W2 += learning_rate * dW2 #Update weights between hidden and output layer
        B1 += learning_rate * dB1 #Update bias at hidden layer
        B2 += learning_rate * dB2 #Update bias at output layer

        #Store previous weight updates for momentum for next iteration
        prev_dW1, prev_dW2 = dW1, dW2
        prev_dB1, prev_dB2 = dB1, dB2
    
    #Typically, learning rate is adjusted by -30% if loss increases and +5% if loss decreases
    if loss < prev_error:
        learning_rate = min(learning_rate * 1.05, max_lr) #Increase learning rate by 5% if loss decreases
    elif loss > prev_error * 1.04:
        learning_rate = max(learning_rate * 0.5, min_lr) #Decrease learning rate by 50% if loss increases
    prev_error = loss #Store previous loss for next iteration

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, RMSE: {rmse:.6f}') #Print RMSE every 100 epochs

#Testing
Z1_test = np.dot(X_test, W1) + B1 #Calculate weighted sum of inputs at hidden layer
A1_test = sigmoid(Z1_test) #Apply sigmoid function to hidden layer
Z2_test = np.dot(A1_test, W2) + B2 #Calculate weighted sum of inputs at output layer
predictions = sigmoid(Z2_test) #Apply sigmoid function to output layer

#Compute error metrics
msre = np.mean(((y_test - predictions) / y_test) ** 2)  
ce = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
#Print results
print("\nModel Evaluation Metrics:")
print(f"Mean Squared Relative Error (MSRE): {msre:.6f}")
print(f"Coefficient of Efficiency (CE): {ce:.6f}")

#Plot actual vs predicted, time series
plt.figure(figsize=(10, 5))
plt.plot(y_test, label=f'Actual {predictand}', linestyle='solid')  
plt.plot(predictions, label=f'Predicted {predictand}', linestyle='dashed')  
plt.xlabel('Day')
plt.ylabel(predictand)  
plt.title(f'Actual vs Predicted {predictand} (Time Series)')  
plt.legend()
plt.show()

#Plot actual vs predicted, scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.xlabel(f'Actual {predictand}')  
plt.ylabel(f'Predicted {predictand}')  
plt.title(f'Actual vs Predicted {predictand} (Scatter Plot)') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.legend()
plt.show()