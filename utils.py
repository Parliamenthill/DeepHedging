import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

def BlackScholes(tau, S, K, sigma, alldata = None):
    d1=np.log(S/K)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2=d1-sigma*np.sqrt(tau)
    price=(S*norm.cdf(d1)-K*norm.cdf(d2))
    delta=norm.cdf(d1) 
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)
    if alldata:
        data = {'npv':price,'delta':delta,'gamma':gamma,'vega':vega,'theta':theta}
        return data
    else:
        return price
    
    
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