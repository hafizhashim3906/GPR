from numpy.linalg import inv

import matplotlib.pyplot as plt
import numpy as np
X_train= np.array([1., 2., 4., 5., 6., 8., 9., 11.])
f = X_train * np.sin(X_train) 
mean = 0
sigma_noise = 0.5
s = np.random.normal(mean, sigma_noise, f.shape)
y = f+s
y_transpose = y.T
X_transpose = X_train.T
X_train = X_train[:,np.newaxis]

n=X_train.shape[0]
sigma_function = 1
lamda = 1
I = np.identity(n)
I= I*(sigma_noise**2)

def kernel(a, b):
    Kernel = (sigma_function**2)*np.exp((-(a-b)**2)/(2.0*lamda**2))
    return Kernel
K_training = kernel(X_train,X_transpose)

from scipy import misc


def partial_derivative(function1, val=0, index=[]):################ ∂/∂θj
    value = index[:]
    def function2(x):
        value[val] = x
        return function1(*value)
    return misc.derivative(function2, index[val], dx = 1e-6)


def log_marginal_likelihood(lamda,sigma_function):
    K_training = (sigma_function**2)*np.exp((-(X_train-X_transpose)**2)/(2.0*lamda**2))
    K_train_noise = K_training + I ############## K(X,X)+ (σ^2)I
    K_train_noise_inv = inv(K_train_noise) ########### [K(X,X)+ (σ^2)I]^-1
    Data_fit1 = (-1.0/2.0)*np.dot(y_transpose,K_train_noise_inv)
    Data_fit = np.dot( Data_fit1,y) ######  -0.5*[yT((K(X,X) +σ2nI)−1)y2]
    Complexity_penalizaion = (-1.0/2.0)* np.log(np.linalg.det(K_train_noise)) ########## -0.5*log|K(X,X) +σ2nI|
    Normalization_constant = (-n/2.0) * np.log(2*np.pi) ######### -0.5*nlog 2π
    log_marginal_likelihood1 = Data_fit + Complexity_penalizaion + Normalization_constant ######## logp(y|X)
    return log_marginal_likelihood1



a = 0.1 ########learning rate
max_iterations = 500
minima = 0.0001 

lamda = 8
sigma_function = 8
lamda_array=[]
sigma_function_array=[]
initial_condition_array=[]

initial_condition = 1000000
i = 0

previous_log_marginal_value = log_marginal_likelihood(lamda,sigma_function)

for i in range(1000000):
    if(initial_condition > minima and i < max_iterations):


        new_lamda = lamda + a * partial_derivative(log_marginal_likelihood, 0, [lamda,sigma_function])#######  θj←θj+α*∂/∂θj[J(θ)]
        new_sigma_function = sigma_function + a * partial_derivative(log_marginal_likelihood, 1, [lamda,sigma_function])####### θj←θj+α*∂/∂θj[J(θ)]

        lamda = new_lamda
        sigma_function = new_sigma_function


        log_marginal_value = log_marginal_likelihood(lamda,sigma_function)
        
        i = i + 1
        initial_condition = abs( previous_log_marginal_value - log_marginal_value )

        previous_log_marginal_value = log_marginal_value
        lamda_array.append(lamda)
        sigma_function_array.append(sigma_function)
        initial_condition_array.append(initial_condition)
        #print(lamda,sigma_function,initial_condition)
initial_condition_Lowest_value_index = np.argmin(initial_condition_array)
lamda_final =(lamda_array[initial_condition_Lowest_value_index])
sigma_function_final = (sigma_function_array[initial_condition_Lowest_value_index])
print('lamda =',lamda_final)
print('sigma_f =',sigma_function_final)