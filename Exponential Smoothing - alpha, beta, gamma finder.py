#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas_gbq
from pylab import rcParams
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import scipy
from scipy import optimize
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# In[2]:


def Box_cox_inverse_index(series_r):
    log_inverse_index=scipy.stats.boxcox(series.values)[1]
    return log_inverse_index


# In[3]:


def series_to_boxcox(series):
    df_input=series.to_frame()
    df_input['Box_cox']=scipy.stats.boxcox(series.values)[0]
    return df_input['Box_cox']


# In[4]:


def boxcox_to_series(boxcox, series_r):
    return scipy.special.inv_boxcox(boxcox, Box_cox_inverse_index(series_r))


# # Classes

# In[5]:


class HoltWinters:
    """Scikit-learn like interface for Holt-Winters method."""

    def __init__(self, season_len=12, alpha=0.5, beta=0.5, gamma=0.5):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.season_len = season_len

    def fit(self, series):
        
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma
        season_len = self.season_len
        seasonals = self._initial_seasonal(series)

        # initial values
        predictions = []
        smooth = series[0]
        trend = self._initial_trend(series)
        predictions.append(smooth)

        for i in range(1, len(series)):
            
            value = series[i]
            previous_smooth = smooth
            seasonal = seasonals[i % season_len]
            smooth = alpha * (value - seasonal) + (1 - alpha) * (previous_smooth + trend)
            trend = beta * (smooth - previous_smooth) + (1 - beta) * trend
            seasonals[i % season_len] = gamma * (value - smooth) + (1 - gamma) * seasonal
            predictions.append(smooth + trend + seasonals[i % season_len])

        self.trend_ = trend
        self.smooth_ = smooth
        self.seasonals_ = seasonals
        self.predictions_ = predictions
        return self
    
    def _initial_trend(self, series):
        season_len = self.season_len
        total = 0.0
        for i in range(season_len):
            total += (series[i + season_len] - series[i]) / season_len

        trend = total / season_len
        
        return trend

    def _initial_seasonal(self, series):
        season_len = self.season_len
        n_seasons = len(series) // season_len

        season_averages = np.zeros(n_seasons)
        for j in range(n_seasons):
            start_index = season_len * j
            end_index = start_index + season_len
            season_average = np.sum(series[start_index:end_index]) / season_len
            season_averages[j] = season_average

        seasonals = np.zeros(season_len)
        seasons = np.arange(n_seasons)
        index = seasons * season_len
        for i in range(season_len):
            seasonal = np.sum(series[index + i] - season_averages) / n_seasons
            seasonals[i] = seasonal

        return seasonals

    def predict(self, n_preds):
        
        predictions = self.predictions_
        original_series_len = len(predictions)
        
        for i in range(original_series_len, original_series_len + n_preds):
            m = i - original_series_len + 1
            prediction = self.smooth_ + m * self.trend_ + self.seasonals_[i % self.season_len]
            predictions.append(prediction)
            
        
#         df1=pd.DataFrame([level,trend,seasonality,predictions], columns =['Level', 'Trend_comp', 'Seasonality_comp'])
#         print (df1)

        return predictions


    def results_split(self,n_preds=4 ):
        predictions = []
        original_series_len = len(predictions)
        level=[]
        trend=[]
        seasonality=[]
        for i in range(original_series_len, original_series_len + n_preds):
            m = i - original_series_len + 1
            prediction = self.smooth + m * self.trend_ + self.seasonals_[i % self.season_len]
            predictions.append(prediction)
            level.append(smooth_)
            trend.append( m * trend_)
            seasonality.append(seasonals_[i % season_len])
        df1=pd.DataFrame([level,trend,seasonality,predictions], columns =['Level', 'Trend_comp', 'Seasonality_comp', 'Predicted_sum'])
        print (df1)
        
    def results_split_2(self):
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma
        season_len = self.season_len
        seasonals = self._initial_seasonal(series)

        # initial values
        predictions = []
        smooth = series[0]
        trend = self._initial_trend(series)
        predictions.append(smooth)

        for i in range(1, len(series)):
            smooth=[]
            seasonality=[]
            trend=[]
            value = series[i]
            previous_smooth = smooth
            seasonal = seasonals[i % season_len]
            smooth = alpha * (value - seasonal) + (1 - alpha) * (previous_smooth + trend)
            trend = beta * (smooth - previous_smooth) + (1 - beta) * trend
            seasonals[i % season_len] = gamma * (value - smooth) + (1 - gamma) * seasonal
            predictions.append(smooth + trend + seasonals[i % season_len])
            smooth.append(smooth)
            trend.append(trend)
            seasonality.append(seasonals[i % season_len])
        df_out=pd.DataFrame(index=series.index, columns=['Actual','Ptredicted','Level','Trend','Seasonality'])
        df_out['Actual']=list(series_r)
        df_out['Predicted']=predictions
        df_out['Level']=smooth
        df_out['Trend']=trend
        df_out['Seasonality']=seasonality
            
            

        self.trend_ = trend
        self.smooth_ = smooth
        self.seasonals_ = seasonals
        self.predictions_ = predictions
        
        return df_out
    
    def fit_2(self, series):
        
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma
        season_len = self.season_len
        seasonals = self._initial_seasonal(series)

        # initial values
        predictions = []
        smooth = series[0]
        trend = self._initial_trend(series)
        predictions.append(smooth)

        for i in range(1, len(series)):
            level=[]
            trend=[]
            seasonality=[]
            value = series[i]
            previous_smooth = smooth
            seasonal = seasonals[i % season_len]
            smooth = alpha * (value - seasonal) + (1 - alpha) * (previous_smooth + trend)
            trend = beta * (smooth - previous_smooth) + (1 - beta) * trend
            seasonals[i % season_len] = gamma * (value - smooth) + (1 - gamma) * seasonal
            level.append(smooth)
            trend.append(trend)
            seasonality.append(seasonal)
            predictions.append(smooth + trend + seasonals[i % season_len])

        self.trend_ = trend
        self.smooth_ = smooth
        self.seasonals_ = seasonals
        self.predictions_ = predictions
        
        df_out=pd.DataFrame(index=series.index, columns=['Actual','Ptredicted','Level','Trend','Seasonality'])
        df_out['Actual']=list(series_r)
        df_out['Predicted']=predictions
        df_out['Level']=level
        df_out['Trend']=trend
        df_out['Seasonality']=seasonality
        
        
        return df_out


# In[6]:


import logging
from typing import Optional

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

LOGGER = logging.getLogger(__name__)


class TimeSeriesSplit_ext(_BaseKFold):  # pylint: disable=abstract-method
    

    def __init__(self,
                 n_splits: Optional[int] = 5,
                 train_size=24,
                 test_size: Optional[int] = None,
                 delay: int = 0,
                 force_step_size: Optional[int] = None):

        if n_splits and n_splits < 5:
            raise ValueError(f'Cannot have n_splits less than 5 (n_splits={n_splits})')
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.train_size = train_size

        if test_size and test_size < 0:
            raise ValueError(f'Cannot have negative values of test_size (test_size={test_size})')
        self.test_size = test_size

        if delay < 0:
            raise ValueError(f'Cannot have negative values of delay (delay={delay})')
        self.delay = delay

        if force_step_size and force_step_size < 1:
            raise ValueError(f'Cannot have zero or negative values of force_step_size '
                             f'(force_step_size={force_step_size}).')

        self.force_step_size = force_step_size

    def split(self, X, y=None, groups=None):
        
        X, y, groups = indexable(X, y, groups)  # pylint: disable=unbalanced-tuple-unpacking
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        delay = self.delay

        if n_folds > n_samples:
            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples: {n_samples}.')

        indices = np.arange(n_samples)
        split_size = n_samples // n_folds

        train_size = self.train_size or split_size * self.n_splits
        test_size = self.test_size or n_samples // n_folds
        full_test = test_size + delay

        if full_test + n_splits > n_samples:
            raise ValueError(f'test_size\\({test_size}\\) + delay\\({delay}\\) = {test_size + delay} + '
                             f'n_splits={n_splits} \n'
                             f' greater than the number of samples: {n_samples}. Cannot create fold logic.')

        # Generate logic for splits.
        # Overwrite fold test_starts ranges if force_step_size is specified.
        if self.force_step_size:
            step_size = self.force_step_size
            final_fold_start = n_samples - (train_size + full_test)
            range_start = (final_fold_start % step_size) + train_size

            test_starts = range(range_start, n_samples, step_size)

        else:
            if not self.train_size:
                step_size = split_size
                range_start = (split_size - full_test) + split_size + (n_samples % n_folds)
            else:
                step_size = (n_samples - (train_size + full_test)) // n_folds
                final_fold_start = n_samples - (train_size + full_test)
                range_start = (final_fold_start - (step_size * (n_splits - 1))) + train_size

            test_starts = range(range_start, n_samples, step_size)

        # Generate data splits.
        for test_start in test_starts:
            idx_start = test_start - train_size if self.train_size is not None else 0
            # Ensure we always return a test set of the same size
            if indices[test_start:test_start + full_test].size < full_test:
                continue
            yield (indices[idx_start:test_start],
                   indices[test_start + delay:test_start + full_test])
            
if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4, 5, 6])
    tscv = TimeSeriesSplit(n_splits=5)
    print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    for train_index, test_index in tscv.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print("---------------------------------------------")
#     LARGE_IDX = np.arange(0, 30)
#     rolling_window = TimeSeriesSplit(train_size=10, test_size=5, delay=3)
#     print(rolling_window)
#     for train_index, test_index in rolling_window.split(LARGE_IDX):
#         print('TRAIN:', train_index, 'TEST:', test_index)
#         X_train, X_test = LARGE_IDX[train_index], LARGE_IDX[test_index]


# # Functions

# In[7]:


def data_cleaner_new(df):
    df['Transaction_date']=pd.to_datetime(df['Transaction_date'])
    df=df.groupby('Transaction_date')[['QTY']].sum().reset_index()
    df = df.set_index('Transaction_date')
    df=df[['QTY']].resample('M').sum()
    df.rename(columns = {'QTY':'Quantity'}, inplace = True) 
    return df


# In[8]:


def timeseries_cv_score(params, series, loss_function, season_len=12, n_splits=5, test_size=6):
    """
    Iterating over folds, train model on each fold's training set,
    forecast and calculate error on each fold's test set.
    """
    errors = []    
    alpha, beta, gamma = params
    time_series_split = TimeSeriesSplit_ext(n_splits=n_splits) 

    for train, test in time_series_split.split(series):
        model = HoltWinters(season_len, alpha, beta, gamma)
       
        model.fit(series[train])

        # evaluate the prediction on the test set only
        predictions = model.predict(n_preds=len(test))
        test_predictions = predictions[-len(test):]
        test_actual = series[test]
        error = loss_function(test_actual, test_predictions)
        errors.append(error)

    return np.mean(errors)


# In[ ]:





# In[9]:


def plot_triple_exponential_smoothing(series, alpha, beta, gamma):
    
    plt.figure(figsize=(20, 8))
    holt_obj=HoltWinters(season_len=12, alpha=alpha, beta=beta, gamma=gamma)
    holt_obj.fit(series)
    results = holt_obj.predict(n_preds=0)
    plt.plot(results, label='Alpha = {}, beta = {}, gamma = {}'.format(alpha, beta, gamma))

    plt.plot(series, label='Actual')
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Triple Exponential Smoothing')
    plt.grid(True)


# In[ ]:





# # Data

# In[10]:


csv2=pd.read_csv('C:/Users/139317/Downloads/US_ECOM_TRAD_FOSSIL_July.csv')
df_us_ecom_trad=pd.DataFrame(csv2)
df_us_ecom_trad


# In[11]:


df_us_sale=data_cleaner_new(df_us_ecom_trad)
df_us_sale. head()


# In[12]:


series=df_us_sale['Quantity']


# In[13]:


df_input=series.to_frame()
df_input['Box_cox']=scipy.stats.boxcox(series.values)[0]
df_input


# In[14]:


inversion_index=Box_cox_inverse_index(series)


# In[15]:


series=df_input['Box_cox']


# In[16]:


series_r=df_input['Quantity']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Graphs

# In[17]:


plt.figure(figsize=(15, 7))
plt.plot(df_us_sale['Quantity'])
plt.title('Sold Quantity')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Determining optimal alpha, beta & gamma parameters

# In[18]:


#provide initial values for model parameters' alpha, beta and gamma
# and leave out the last 5 points of our time series as test set
x = [0, 0, 0]
test_size = 5
data = series.values[:-test_size]
opt = scipy.optimize.minimize(timeseries_cv_score, x0=x, 
               args=(data, mean_squared_log_error), 
               method='TNC', bounds=((0, 1), (0, 1), (0, 1)))

print('original parameters: {}'.format(str(x)))
print('best parameters: {}'.format(str(opt.x)))


# In[ ]:





# # Fitting Holt Winter Object with the above parameters

# In[19]:


HW=HoltWinters(season_len=12, alpha=opt.x[0], beta=opt.x[1], gamma=opt.x[2])


# In[20]:


HW.fit(series)


# In[21]:


pred_0=HW.predict(n_preds=0)


# In[22]:


#Boxcox transformed predictions for 4 periods into future
pred_4=HW.predict(n_preds=4)


# In[23]:


pred_qty_4=scipy.special.inv_boxcox(pred_4, inversion_index)
pred_qty_4=pred_qty_4.astype(int)
pred_qty_4


# In[24]:


series_r
df_pred_vs_act=pd.DataFrame(index=series_r.index, columns=['Actual','Predicted', 'Percentage_diff'])
df_pred_vs_act['Actual']=series_r
df_pred_vs_act['Predicted']=list(pred_qty_4[:-4])
df_pred_vs_act['Percentage_diff']=(abs(df_pred_vs_act['Predicted']-df_pred_vs_act['Actual']))*100/df_pred_vs_act['Actual']


# In[25]:


df_pred_vs_act


# In[26]:


len(series_r)


# In[27]:


len(pred_4)


# In[28]:


HW.fit_2(series)


# In[29]:


HW.results_split_2()


# In[30]:


HW.seasonals_


# In[31]:


scipy.special.inv_boxcox(HW.seasonals_,inversion_index)


# In[32]:


scipy.special.inv_boxcox(scipy.special.inv_boxcox(HW.seasonals_,inversion_index), inversion_index)


# In[33]:


HW.results_split()


# In[27]:


len(series)


# In[28]:


len(pred_4)


# In[29]:


g=plt.scatter(df_pred_vs_act['Actual'], df_pred_vs_act['Predicted'])
g.axes.set_yscale('log')
g.axes.set_xscale('log')
g.axes.set_xlabel('True Values ')
g.axes.set_ylabel('Predictions ')
g.axes.axis('equal')
g.axes.axis('square')


# In[30]:



plt.plot(df_pred_vs_act.Actual)
plt.plot(df_pred_vs_act.Predicted, 'o')

plt.show()


# In[31]:


def plot_double_exponential_smoothing(series, alphas, betas):
    """Plots double exponential smoothing with different alphas and betas."""    
    plt.figure(figsize=(20, 8))
    for alpha, beta in zip(alphas, betas):
        results = double_exponential_smoothing(series, alpha, beta)
        plt.plot(results, label='Alpha {}, beta {}'.format(alpha, beta))

    plt.plot(series, label='Actual')
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Double Exponential Smoothing')
    plt.grid(True)

plot_double_exponential_smoothing(series_r.values, alphas=[0.9, 0.9], betas=[0.1, 0.9])


# In[32]:


plt.plot(df_pred_vs_act.Actual, df_pred_vs_act.Predicted, label='Actual')
plt.legend(loc='best')
plt.axis('tight')
plt.title('Triple Exponential Smoothing')
plt.grid(True)


# In[33]:


plt.plot('Predicted', 'Actual', data=df_pred_vs_act)


# In[ ]:





# In[ ]:




