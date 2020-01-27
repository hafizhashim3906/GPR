import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

X_train= np.array([1., 2., 4., 5., 6., 8., 9., 11.]) ########## Random input samples
f = X_train * np.sin(X_train) ########## Function to produce outputs (observations)
X_train = X_train[:,np.newaxis]
X_transpose = X_train.T

mean = 0
sigma_noise = 0.5
n=X_train.shape[0]
sigma_function = 4
lamda = 1
I = np.identity(n)
I= I*(sigma_noise**2)
s = np.random.normal(mean, sigma_noise, f.shape)########################## Noise function 

y = f+s ############## Adding noise to the function
y_transpose = y.T
plt.plot(X_train, y,'bo')
plt.grid(True, color='#666666', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training samples,Test samples and Predicted function', fontsize=10)
plt.show


##################### Squared exponenetial kernel
def kernel(a, b):
    Kernel = (sigma_function**2)*np.exp((-(a-b)**2)/(2.0*lamda**2))
    return Kernel
########################################################################
############## Making covariance matrix
K_training = kernel(X_train,X_transpose) ######### K(X,X) 
K_train_noise = K_training + I ############## K(X,X)+ (σ^2)I
K_train_noise_inv = inv(K_train_noise) ########### [K(X,X)+ (σ^2)I]^-1
#K_train_test = kernel(X_train,X_test) ################# K(X*,X)
#K_train_test_transpose = K_train_test.T
#K_test = kernel(X_test,X_test) ############### K(X*,X*)
#############################################################################

##########################################################################

########################################################################  Testing
X_test1 = [3., 7., 12] ############ 3 test samples
y_test_mean_array1 = []
y_test_var_array1 = []

for X_test in X_test1:
    
    K_train_test = kernel(X_train,X_test) ################# K(X*,X)
    K_train_test_transpose = K_train_test.T
    K_test = kernel(X_test,X_test) ############### K(X*,X*)
    mean_test = np.dot(K_train_test_transpose,K_train_noise_inv) ###### K(X*,X)[K(X,X)+ (σ^2)I]^-1
    y_test_var = K_test - (np.dot(K_train_test_transpose, np.dot(K_train_noise_inv,K_train_test)))###### cov = K(X*,X*)-[[K(X*,X)[K(X,X)+ (σ^2)I]^-1]K(X,X*)]
    y_test_mean2 = np.dot(mean_test,y) ############# mean = [K(X*,X)[K(X,X)+ (σ^2)I]^-1]y
    y_test_mean_array1.append(y_test_mean2)
    y_test_var2 = K_test - (np.dot(K_train_test_transpose, np.dot(K_train_noise_inv,K_train_test)))###### cov = K(X*,X*)-[[K(X*,X)[K(X,X)+ (σ^2)I]^-1]K(X,X*)]
    y_test_var_array1.append(y_test_var2)
    #print(y_test_var2[0])
    #plt.errorbar(X_test,y_test_mean2[0],y_test_var2[0])

#####################################################
########################################################### Predicion of function
X_test2 = np.arange(0,13,0.1)
y_test_mean_array2 = []
y_test_var_array2= []

for X_test in X_test2:
    
    K_train_test = kernel(X_train,X_test) ################# K(X*,X)
    K_train_test_transpose = K_train_test.T
    K_test = kernel(X_test,X_test) ############### K(X*,X*)
    mean_test = np.dot(K_train_test_transpose,K_train_noise_inv) ###### K(X*,X)[K(X,X)+ (σ^2)I]^-1
    y_test_var = K_test - (np.dot(K_train_test_transpose, np.dot(K_train_noise_inv,K_train_test)))###### cov = K(X*,X*)-[[K(X*,X)[K(X,X)+ (σ^2)I]^-1]K(X,X*)]
    y_test_mean2 = np.dot(mean_test,y) ############# mean = [K(X*,X)[K(X,X)+ (σ^2)I]^-1]y
    y_test_mean_array2.append(y_test_mean2)
    y_test_var2 = K_test - (np.dot(K_train_test_transpose, np.dot(K_train_noise_inv,K_train_test)))###### cov = K(X*,X*)-[[K(X*,X)[K(X,X)+ (σ^2)I]^-1]K(X,X*)]
    y_test_var_array2.append(y_test_var2)
    m1 = y_test_mean2[0]
    y_err = y_test_var2[0]
    plt.errorbar(X_test,y_test_mean2[0],y_test_var2[0])


#plt.plot(X_train, y,'bo')
plt.plot(X_test1, y_test_mean_array1,'ro', label = 'test_samples')
plt.plot(X_test2,y_test_mean_array2,'--', label = 'Predicted function')
plt.legend()
plt.grid(True, color='#666666', linestyle='--')
