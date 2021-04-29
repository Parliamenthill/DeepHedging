





class DeepHedge:
    def __init__(self, N, T, S0, strike):
        self.N = N # time disrectization    
        self.S0 = S0 # initial value of the asset
        self.T = T # maturity
        self.dt = T / N
        self.strike = strike
        self.payoff_function = lambda x : 0.5*(np.abs(x-self.strike)+x-self.strike) # European call option payoff
        self.option_type = 'eurocall'
        self.strike_modified = 1.3
        
        if po == np.inf:
            self.payoff_function_modified = lambda x : 0.5*(np.abs(x-self.strike_modified)+x-self.strike_modified) 
            def ploss(payoff, outputs):
                loss = tf.math.reduce_max(tf.nn.relu(payoff - outputs),keepdims = True)
                return loss
        if po == 1:
            self.payoff_function_modified = lambda x : (x-self.strike) * (x > self.strike_modified) 
            def ploss(payoff, outputs):
                loss = (tf.nn.relu(payoff - outputs))**po
                return loss
        if po == 0:
            self.payoff_function_modified = lambda x : (x-self.strike) * ((x < self.strike_modified) & (x > self.strike))
            def ploss(payoff, outputs):
                loss = tf.math.sign(tf.nn.relu(payoff - outputs))
                return loss
        self.ploss = ploss

    def generate_data(self):        
        self.m = 1 # dimension of price
        self.mu = 0.02
        self.sigma = 0.3
        self.Ktrain = 10**5
        self.price_path, self.time_grid = utils_efficient.simulate_GBM(self.m,self.Ktrain,self.N,self.T,\
                                                             self.mu,self.sigma,self.S0, 'equi')
        self.price_path_EMM, _ = utils_efficient.simulate_GBM(self.m,self.Ktrain,self.N,self.T,\
                                                             0,self.sigma,self.S0, 'equi')
        self.payoff = self.payoff_function(self.price_path[:,-1]) 
        self.payoff_modified = self.payoff_function_modified(self.price_path[:,-1]) 
        if self.po == np.inf:
            self.delta_output, self.delta_path, self.option_path = utils_efficient.delta_hedge(self.price_path,self.payoff_modified,self.T,self.strike_modified,self.sigma,self.option_type,self.time_grid)
            self.initial_wealth,_ = utils_efficient.BlackScholes(self.T, self.S0, self.strike_modified, self.sigma, self.option_type)
        else:
            self.delta_output, self.delta_path, self.option_path = utils_efficient.delta_hedge(self.price_path,self.payoff_modified,self.T,\
                                                                                               [self.strike, self.strike_modified],self.sigma,self.po,self.time_grid)
#             self.initial_wealth = self.payoff_function_modified(self.price_path_EMM[:,-1]).mean() ###### EMM
            self.initial_wealth = self.option_path[0,0,0]
            
    def plot_payoff(self):
        print(f"real premium: {self.initial_wealth:{1}.{4}}")           # real premium
        f,p = plt.subplots(1,2, figsize = [10,3], sharey = True)
        p[0].scatter(self.price_path[:,-1,0], self.payoff[:,0], s = 1, alpha = 0.5, label = 'payoff')
        p[1].scatter(self.price_path[:,-1,0], self.payoff_modified[:,0], s = 1, alpha = 0.5, label = 'modified payoff')
        p[0].grid()
        p[1].grid()
        plt.show()
    def build_model(self):
        self.model_hedge, self.Network0, self.Networks = build_dynamic(self.m, self.N, False, False, self.initial_wealth, self.po, self.ploss)
    def prepare_data(self):    
        control_path = False
        if not control_path:
            self.split = int(self.Ktrain/2)
            self.xtrain = [self.price_path[:self.split], self.payoff[:self.split]]  # input be price_path
            self.ytrain = self.payoff[:self.split]*0  # output be payoff
            self.xtest = [self.price_path[self.split:], self.payoff[:self.split]]  # input be price_path
            self.ytest = self.payoff[self.split:]*0  # output be payoff    
        else:
            self.split = int(self.Ktrain/2)
            self.xtrain = [self.price_path[:self.split], self.payoff[:self.split], self.option_path[:self.split]]  # input be price_path
            self.ytrain = self.payoff[:self.split]*0  # output be payoff
            self.xtest = [self.price_path[self.split:], self.payoff[:self.split], self.option_path[self.split:]]  # input be price_path
            self.ytest = self.payoff[self.split:]*0  # output be payoff    
    def train(self):
        def zeroloss(y_true, y_predict):
            return tf.reduce_sum(y_predict*0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # specify the optmizer 
        # model_hedge.compile(optimizer = optimizer,loss='mse') # specify the mean square loss 
        self.model_hedge.compile(optimizer = optimizer,loss=zeroloss) # specify the expected shortfall
        self.model_hedge.fit(x=self.xtrain,y=self.ytrain, epochs=10,verbose=True,batch_size=256) # train the model
    def predict(self):
        self.hedge_output_train = self.model_hedge.predict(self.xtrain) # compute the output (deep replicate payoff) with trained model 
        self.hedge_output_test = self.model_hedge.predict(self.xtest) # compute the output (deep replicate payoff) with trained model
        print('train: ',tf.reduce_mean(self.ploss(self.payoff[:self.split], self.hedge_output_train)).numpy())
        print('test: ',tf.reduce_mean(self.ploss(self.payoff[self.split:], self.hedge_output_test)).numpy())
        print('best: ',tf.reduce_mean(self.ploss(self.payoff, self.delta_output)).numpy())
        
        f,p = plt.subplots(1,3, figsize = [20,5], sharey = True)
        p[0].scatter(self.price_path[self.split:,-1,0], self.hedge_output_test[:,0], s = 1, alpha = 0.5, label = 'deep hedge test ')   # deep replicate payoff 
        p[1].scatter(self.price_path[:self.split,-1,0], self.hedge_output_train[:,0], s = 1, alpha = 0.5, label = 'deep hedge train')   # deep replicate payoff 
        if self.po == np.inf:
            p[2].scatter(self.price_path[:,-1,0], self.delta_output[:,0], s = 1, alpha = 0.5, label = 'delta hedge')   # delta replicate payoff 
        if self.po == 0:
            p[2].scatter(self.price_path[:,-1,0], self.delta_output[:,0], s = 1, alpha = 0.5, label = 'delta hedge')   # delta replicate payoff 
        
        for i in range(3):
            p[i].scatter(self.price_path[:,-1,0], self.payoff[:,0], s = 1, alpha = 0.5, label = 'real payoff')        # real payoff
            p[i].scatter(self.price_path[:,-1,0], self.payoff_modified[:,0], s = 1, alpha = 0.5, label = 'modified payoff')
            
            p[i].legend()
            p[i].grid()
    def compare_strategy(self):
        f,p = plt.subplots(1,5,figsize = [20,5])
        for i in range(5):
            n = 20*i + 10
            pr = np.linspace(0.5,2,100)[:,None]  # tf.tensor of different price 

            he = self.Networks[n](tf.math.log(pr)) # the stategy network 
            p[i].plot(pr[:,0],he[:,0], label = 'deep hedge') # plot the relation between price and deep strategy
            if self.po == np.inf:
                _ , delta = utils_efficient.BlackScholes(self.T - self.time_grid[n], pr, self.strike_modified, self.sigma, self.option_type)
                p[i].plot(pr, delta, label = 'delta hedge') # plot the relation between price and delta strategy
            if self.po == 0:
                _ , delta = utils_efficient.BS0(self.T - self.time_grid[n], pr, [self.strike,self.strike_modified], self.sigma, self.option_type)
                p[i].plot(pr, delta, label = 'delta hedge') # plot the relation between price and delta strategy
            p[i].title.set_text(f"At time: {self.time_grid[n]:{1}.{4}}")
            p[i].legend()
            p[i].grid()
        plt.show()
        