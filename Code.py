
# coding: utf-8

# In[94]:

import sklearn #import kneighborsRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')


# In[95]:

literacy = pd.read_csv("YIP.csv" )
literacy.reset_index(drop=True, inplace=True)


# In[96]:

literacy['Youth illiterate population, 15-24 years, both sexes'] = literacy['Youth illiterate population, 15-24 years, both sexes'].apply(lambda x: float(x.split()[0].replace(',', '')))

y = literacy['Youth illiterate population, 15-24 years, both sexes']
X = literacy.iloc[:,0]
#X =  X.reshape(-1, 1)


# In[102]:

x_train , x_test , y_train , y_test = train_test_split(X , y)
#y_train


# In[98]:

x_train = x_train.reshape(-1,1)


# In[99]:

linreg = LinearRegression().fit(x_train , y_train)
#X = X.astype(float)


# In[104]:

plt.figure(figsize = (5,4))
plt.scatter(X , y ,marker = 'o' , s = 50 , alpha = 0.8)
plt.plot(X , linreg.coef_*X+linreg.intercept_ , 'r-')
plt.title('illiteracy data between the age of 15 - 24')
plt.xlabel('Years')
plt.ylabel('illiterate people no(million)')
plt.show()


# In[ ]:



