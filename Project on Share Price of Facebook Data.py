#!/usr/bin/env python
# coding: utf-8

# # Project on Daily stock price of Facebook from 31-12-2014 to 05-02-2018

# In[1]:


#Hello Everyone
#This is Umang!
#I have two datasets of historical stock data of Microsoft and Facebook 
#I am going to analyse the data and view basic trends using python


# In[2]:


#Pandas is used for data structure for time series data
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import norm
from math import sqrt


# In[3]:


import matplotlib. pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Analyzing Facebook Data

# In[4]:


#Importing Facebook data, that we uploaded in our jupiter notebook
fb = pd.read_csv(r"C:\Users\umang\Desktop\facebook.csv")


# In[5]:


#To obtain first five rows and check the heads of our data
fb.head()


# In[6]:


#This data shows Opening price, closing price, highest price, lowest price, Adjust close with is Stock closing price, and volume
#Lets attract some more basic details of our data in order to study it
fb.shape


# In[7]:


fb.columns


# In[8]:


fb.describe()


# In[9]:


#Plotting the closing price 
plt.figure(figsize=(10,8))
fb['Close'].plot()
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.title("Closing Price for Facebook")
plt.show()


# In[10]:


#We can notice from the graph that the growth of closing stock price was low till 2016. But it rose speadily after Dec 2016.


# In[11]:


#Plotting year wise scatter plot
plt.scatter(fb['Year'],fb['Close'])
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.title("Closing Price for Facebook")
plt.show()


# In[12]:


fb_2015 = fb.loc[fb.Year==2015]
mean_fb_2015 = fb_2015['Close'].mean()


fb_2016 = fb.loc[fb.Year==2016]
mean_fb_2016 = fb_2016['Close'].mean()

      
fb_2017 = fb.loc[fb.Year==2017]
mean_fb_2017 = fb_2017['Close'].mean()

      
fb_2018 = fb.loc[fb.Year==2018]
mean_fb_2018 = fb_2018['Close'].mean()


# In[13]:


X = ["2015","2016","2017","2018"]
Y = [mean_fb_2015, mean_fb_2016, mean_fb_2017, mean_fb_2018]


# In[14]:


plt.bar(X,Y)
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.title("Closing Price for Facebook")


# In[15]:


#Changing the date into datetime format
fb['Date'] = pd.to_datetime(fb.Date)
fb.head()


# In[16]:


#creating a new column in dataframe for year
fb['Year'] = pd.DatetimeIndex(fb.Date).year
fb.head()


# In[17]:


fb['Price1'] = fb['Close'].shift(-1)
fb.head()


# In[18]:


#creating new column of price difference (close price of tomorow - close price of today)
fb['PriceDiff']= fb['Price1'] - fb['Close']
fb.head()


# In[19]:


#creating a column of daily return
fb['Return'] = fb['PriceDiff']/fb['Close']
fb.head()


# In[20]:


#Finding the direction of the price difference
fb['Direction'] = [1 if fb.loc[ei, 'PriceDiff']>0 else -1 for ei in fb.index]
fb.head()


# In[21]:


#calculate moving average of close price over three days
fb['Aveerage3']= (fb['Close'] + fb['Close'].shift(1) + fb['Close'].shift(2))/3
fb.head()


# In[22]:


fb['MA40'] = fb['Close'].rolling(40).mean()
fb['MA200'] = fb['Close'].rolling(200).mean()
fb.head()


# In[23]:


#MA40 is a fast signal for recent changes(short term) and MA200 is a slow signal for long term changes  
#when MA40>MA200 Traders beleive it that stock price will go upwards for a while 
#ie when MA40>MA200 we buy and hold one share
fb['Close'].plot()
fb['MA40'].plot()
fb['MA200'].plot()


# In[24]:


fb['MA40']= fb['Close'].rolling(40).mean()
fb['MA200']= fb['Close'].rolling(200).mean()
fb = fb.dropna()


# In[25]:


#creating a column of shares
fb['Shares']=[1 if fb.loc[ei, 'MA40']> fb.loc[ei, 'MA200'] else 0 for ei in fb.index]
fb.head()


# In[26]:


#we already have a new column of Price1 ie close price for tomorrow
#Now we create a column for profit
fb['Profit']= [fb.loc[ei,'Price1']-fb.loc[ei,'Close']
              if fb.loc[ei,'Shares']==1
              else 0 for ei in fb.index]
fb.head()


# In[27]:


#Plotting the proft
fb['Profit'].plot()


# In[28]:


#Finding the cumulative profit or wealth
fb['Wealth']= fb['Profit'].cumsum()
fb['Wealth'].plot()


# In[29]:


print("Total money win is ",fb.loc[fb.index[-2],'Wealth'])


# In[30]:


print("Total money spent is ",fb.loc[fb.index[0],'Close'])


# # Estimating the Average stock return with 90% Confidence Interval
# #We use Log return 

# In[31]:


fb['Log_Return'] = np.log(fb['Close'].shift(-1)) - np.log(fb['Close'])
fb.head()


# In[32]:


#Getting Sample size, Sample Mean, Sample standard deviation
sample_size = fb['Log_Return'].shape[0]
sample_mean = fb['Log_Return'].mean()
sample_std = fb['Log_Return'].std(ddof=1) / sample_size**5
z_score = 1.645

print("Sample Size =",sample_size)
print("Sample Mean =",sample_mean)
print("Sample Standard Deviation =",sample_std)
print("Z-Score =",z_score)


# In[33]:


# Let Margin of Error be z_score ×sqrt(sample_mean×(1−sample_mean)/sample_size)
margin_of_error =z_score * sqrt(sample_mean*(1 - sample_mean)/sample_size)
margin_of_error


# In[34]:


interval_left = sample_mean - margin_of_error
interval_right = sample_mean + margin_of_error


# In[35]:


#90% confidence interval tells that there is 90% chance that average stock return lies between interval left and interval right
print("90% confidence interval is ", (interval_left, interval_right))

