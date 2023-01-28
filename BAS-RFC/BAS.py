import numpy as np
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
import pandas as pd


# np.random.seed(0)

def load_data():
    # Load structured data
    # Fill in data and labels here
    # data = 
    # target = 
    # Recursive feature elimination can replace other feature engineering algorithms
    # est = RFC()
    # rfe = RFE(estimator=est, n_features_to_select=12)
    # rfe.fit(data, target)
    # feature_num = rfe.support_
    # data = data[:, feature_num]
    # return data, target
    pass


class BAS():

    def __init__(self, cmo=0.1, global_iter=6, iter=9, stop_percent=0.04, step=0.1, d0=0.1, min_step=0.00001, k=3):
        super(BAS, self).__init__()
        self.c_mo = cmo               # Step decay factor
        self.cc_mo = 1-self.c_mo       # Influence factor
        self.global_iter = global_iter          # Number of global Iterations
        self.iter = iter                 # Number of single iteration
        self.stop_percent = stop_percent       # When the number of single iterations accounts for more than stop_percent times of the total number of iterations, the current iteration will be stopped if the objective function value is the same
        self.step = step                # Initial search step
        self.d0 = d0                 # Tentacle spacing
        self.min_step = min_step           # Minimum step size
        self.k = k                     # Variable dimension
        # Step decay factor, global iteration number, word iteration number, stop standard, initial search step length, whisker spacing, and array formed by minimum step length
        self.eval_ls = []

        self.sol_bound = np.array([
            [1, 201], [1, 21], [1, 51]
            ])                    # Hyperparametric search space
        self.sol_minus = self.sol_bound[:, 1]-self.sol_bound[:, 0]
        self.eval_tab = pd.DataFrame(
            columns=[
                'func_value',
                'tolerance',
                'nest', 'mdep', 'msl'
                ]) 
    
    @staticmethod
    def normalize(x)->np.float: # Unitized vector
        x_mod = np.linalg.norm(x)
        return x / x_mod

    def f(self, solution):            # Test function (the objective is to obtain the maximum value of the function, and the extreme value of the function is 1)
        (n_estimators, max_depth, min_samples_leaf) = (self.sol_minus*np.array(solution)+self.sol_bound[:, 0]).astype(int)
        est = RFC(n_estimators=n_estimators, max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf, random_state=0, class_weight='balanced')
        est.fit(self.X_train, self.y_train)
        score = est.score(self.X_test, self.y_test)
        return score

    def optim(self, data, target):
        # If the suspension criteria have been met and no improvement has been made, stop the current iteration
        stop_criteria = self.stop_percent*self.iter 
        # Generate training&test dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.1, random_state=0)
        # Global iterative process
        for _ in range(self.global_iter):
            x = np.random.rand(self.k)     # Generate centroid coordinates of longicorn at random
            fx = self.f(x)
            xl = x                    # Left tentacle coordinate
            xr = x                    # Right tentacle coordinate
            for _ in range(self.iter):   # Start individual iteration of single Taurus
                his_fx = fx
                if np.any(fx==self.eval_tab.func_value):
                    self.eval_tab.loc[self.eval_tab['func_value']==fx, 'tolerance'] += 1
                    self.eval_tab.loc[self.eval_tab['func_value']==fx, 'nest'] = x[0]
                    self.eval_tab.loc[self.eval_tab['func_value']==fx, 'mdep'] = x[1]
                    self.eval_tab.loc[self.eval_tab['func_value']==fx, 'msl'] = x[2]
                else:
                    self.eval_tab = self.eval_tab.append({
                        'func_value':fx, 'tolerance':1, 
                        'nest':x[0], 'mdep':x[1], 'msl':x[2]
                        }, ignore_index=True)
                max_tolerance = np.max(self.eval_tab.tolerance)
                if max_tolerance > stop_criteria:
                    break 
                dir = np.random.rand(self.k)
                dir = self.normalize(dir)
                tmp_d = self.d0*dir/2
                xl = np.clip(x+tmp_d, 0, 1)
                xr = np.clip(x-tmp_d, 0, 1)
                fl = self.f(xl)
                fr = self.f(xr)
                x = np.clip(x+self.step*dir*np.sign(fl-fr), 0, 1)
                now_fx = self.f(x)
                # Change the step size and ensure that the step size is greater than the minimum step size
                self.step = np.clip(self.step*self.c_mo+np.abs(his_fx-now_fx)*self.cc_mo, a_min=self.min_step, a_max=1) 
            mfv = np.max(self.eval_tab.func_value)
            msol = self.eval_tab.loc[self.eval_tab['func_value']==mfv, ['nest', 'mdep', 'msl']].values[0]
            self.eval_ls.append([mfv, msol])
            self.eval_tab = self.eval_tab.drop(index=self.eval_tab.index)
        tmp_max = 0
        for i in self.eval_ls:
            if i[0] > tmp_max:
                max_sol = i[1]
                tmp_max = i[0]
        self.max_sol = max_sol
        
        return tmp_max
    
    def predict(self):
        (n_estimators, max_depth, min_samples_leaf) = (self.sol_minus*np.array(self.max_sol)+self.sol_bound[:, 0]).astype(int)
        rfc = RFC(n_estimators=n_estimators, max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf, random_state=0, class_weight='balanced')
        rfc.fit(self.X_train, self.y_train)
        y_pred = rfc.predict(self.X_test).astype(int)
        return y_pred, self.y_test

if __name__ == '__main__':
    X, y = load_data()
    t1 = time.time()
    bas = BAS(cmo=0.1, global_iter=6, iter=9, stop_percent=0.04, step=0.1, d0=0.1, min_step=0.00001)

    accu = bas.optim(X, y)
    t2 = time.time()
    print(accu, t2-t1)
        # y_pred = bas.predict()
