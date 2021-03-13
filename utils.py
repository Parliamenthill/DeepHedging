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