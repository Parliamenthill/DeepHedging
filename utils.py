import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
keras = tf.keras

def BlackScholes(tau, S, K, sigma, option_type):
    d1=np.log(S/K)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2=d1-sigma*np.sqrt(tau)
    delta=norm.cdf(d1) 
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)
    if option_type == 'eurocall':
        price = (S*norm.cdf(d1)-K*norm.cdf(d2))
        hedge_strategy = delta
    elif option_type == 'eurodigitalcall':
        price = norm.cdf(d2)
        hedge_strategy = gamma
    return price, hedge_strategy
    
def stimulate_Heston(m,Ktrain,N,T,corr,kappa,theta,xi,S0,v0):
    S = [S0]
    v = [v0]
    v_curr = v0; S_curr = S0
    for i in range(N):
        # Simulate standard normal random variables
        Z1_i = np.random.normal(0,1,(Ktrain,m))
        Z2_i = np.random.normal(0,1,(Ktrain,m))
        # Generate correlated Brownian motions
        WS_curr = np.sqrt(T/N)*Z1_i
        Wv_curr = np.sqrt(T/N)*(corr*Z1_i + np.sqrt(1-corr**2)*Z2_i)
        # Calculate volatility and stock price
        # Adjustment: np.abs(v_curr) for np.sqrt() in the last term
#         v_new = v_curr + kappa*(theta-v_curr)*T/N + xi*np.sqrt(np.abs(v_curr))*Wv_curr
        v_new = v_curr + kappa*(theta-v_curr)*T/N + xi*np.sqrt(v_curr)*Wv_curr # without abs 
        S_new = S_curr * np.exp(- v_new**2/2*T/N + v_new*WS_curr)
        v_curr = v_new; S_curr = S_new
        # Append the results to the arrays of each day's value
        v += [v_new]; S += [S_new]
        price_path = np.swapaxes(np.array(S),0,1)
        vol_path = np.swapaxes(np.array(v),0,1)
    return price_path, vol_path


def stimulate_GBM(m,Ktrain,N,T,sigma,S0):
    time_grid = np.linspace(0,T,N+1)
    dt = T/N
    BM_path_helper = np.cumsum(np.random.normal(size = [Ktrain,N,m], loc=0, scale=np.sqrt(dt)),axis = 1) # generate and sum the increment of BM
    BM_path = np.concatenate([np.zeros([Ktrain,1,m]),BM_path_helper],axis = 1) # set initial position of BM be 0 
    price_path = S0 * np.exp(sigma * BM_path - 0.5 * sigma **2 * time_grid[None,:,None])  # from BM to geometric BM
    return price_path
    
    

def build_network(m, n, d, N):
# architecture is the same for all networks
    Networks = []
    trainable = True
    for j in range(N):
        inputs = keras.Input(shape=(m,))
        x = inputs
        for i in range(d):
            if i < d-1:
                nodes = n
                layer = keras.layers.Dense(nodes, activation='tanh',trainable=trainable,
                          kernel_initializer=keras.initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                          bias_initializer='random_normal',
                          name=str(j) + 'step' + str(i) + 'layer')
                x = layer(x)
            else:
                nodes = m
                layer = keras.layers.Dense(nodes, activation='linear', trainable=trainable,
                              kernel_initializer=keras.initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                              bias_initializer='random_normal',
                              name=str(j) + 'step' + str(i) + 'layer')
                outputs = layer(x)
                network = keras.Model(inputs = inputs, outputs = outputs)
                Networks.append(network)
    return Networks


def delta_hedge(price_path,T,K,sigma,option_type):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
    time_grid = np.linspace(0,T,N+1)
    
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    hedge = np.zeros_like(price[:,0,:])  
    premium,_ = BlackScholes(T-time_grid[0], price[:,0,:], K, sigma, option_type)
    for j in range(N):
        _,strategy = BlackScholes(T-time_grid[j],price[:,j,:],K,sigma,option_type)  
        hedge = hedge + strategy * price_difference[:,j,:]   
    outputs = premium + hedge 
    return outputs 
    
    

