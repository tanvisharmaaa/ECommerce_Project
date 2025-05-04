#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 


# In[2]:


df = pd.read_csv('data.csv', encoding='unicode_escape')


# In[3]:


df.head()


# In[4]:


df.info()


# # Business Questions
# 
# 1. What percent of orders get returned (overall & by country)
# 2. Which product cartegory has the highest rate of return?
# 3. Are higher priced items more likely to be returned?
# 4. Do returns differ by month/season?
# 5. Do repeat customers retrun more/less than new customers? 

# # What percent of orders get returned (overall & by country)

# In[5]:


# Cancelled orders 

df['IsReturn'] = df['InvoiceNo'].str.startswith('C')


# In[6]:


df[df['Quantity']<0].head()


# In[7]:


# check for Duplicates 

df.duplicated().sum()


# In[8]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[9]:


#Drop duplicates

df = df.drop_duplicates()


# In[10]:


# Q1 a - Overall Return Rate

total_orders = len(df)
total_returns = df['IsReturn'].sum()

return_rate = total_returns/total_orders *100

print('Overall Return Rate is %f'%return_rate)


# In[11]:


# Q1-b Return Rate by Country 

country_return = df.groupby('Country')['IsReturn'].mean()*100

country_return = country_return.sort_values(ascending=False)
country_return


# #### While the overall return rate is 1.7%, certain countries (e.g., USA: 38.5%, Czech Republic: 16.7%) have significantly higher return rates, suggesting regional return policies, customer behavior, or fulfillment challenges might be influencing this pattern.

# # Which product cartegory has the highest rate of return?

# In[12]:


#Q2: Return Rate by Product Description

df['Description'].unique().shape


# In[13]:


df.shape


# In[14]:


all_words = ' '.join(df['Description'].dropna())

words = all_words.split()

word_counts = Counter(words)

#word_counts.most_common(150)


# In[15]:


pattern = r'(BAG|BOX|MUG|HOLDER|T-LIGHT|PAPER|SIGN|CARD|DECORATION|FRAME|TISSUES|CANDLE|LIGHT|JAR|CUP|PLATE|TREE|BOWL|CLOCK|DOORMAT|BUNTING|DRAWER|LUNCH|GARDEN|CUSHION|TIER|PLASTERS|TRINKET|TRAVEL|CANDLES|RIBBONS|EGG|TINS|COVER|HOOK)'


# In[16]:


df['Category'] = df['Description'].str.extract(pattern, expand=False)


# In[22]:


#df


# In[18]:


#df['Category'].value_counts()


# In[19]:


category_orders = df.groupby('Category').size()
category_returns = df[df['IsReturn']].groupby('Category').size()

category_return_rate = (category_returns / category_orders) * 100
category_return_rate = category_return_rate.fillna(0).sort_values(ascending=False)

category_return_rate


# In[21]:


category_return_rate.plot(kind='barh', figsize=(10,8))
plt.xlabel('Return Rate (%)')
plt.title('Return Rate by Product Category')
plt.show()


# Looks like TIER, JAR, TRINKET have the highest return rates (~4–7%), while categories like RIBBONS and TISSUES are returned <1%.

# # Are higher priced items more likely to be returned?

# In[39]:


# Q3 
bins = [0,5,10,20,50,100,500,1000,2000,5000]

labels = ['0-5','5-10','10-20','20-50','50-100','100-500','500-1000','1000-2000','2000-5000']


# In[40]:


price_df = df[df['UnitPrice'] > 0].copy()


# In[41]:


price_df['PriceRange'] = pd.cut(price_df['UnitPrice'],bins = bins, labels = labels)


# In[42]:


price_orders = price_df.groupby('PriceRange').size()
price_returns = price_df[price_df['IsReturn']].groupby('PriceRange').size()

price_return_rate = (price_returns/price_orders)*100
price_return_rate = price_return_rate.fillna(0).sort_index()

print(price_return_rate)


# In[44]:


price_return_rate.plot(kind='bar')
plt.ylabel('Return Rate (%)')
plt.title('Return Rate by Price Range')
plt.show()


# Our analysis shows a clear correlation between product price and return rate.
# While low-priced items (under 50 USD) maintain a return rate below 5 percent, higher-priced items exhibit significantly elevated return rates: 25% for 50–100 USD, 28 percent for 500–1000 USD, and exceeding 40% for products above $1000.This suggests a need to review our return policy, fraud controls, or customer satisfaction measures for high-value transactions.

# # Do returns differ by month/season?

# In[45]:


df['Month'] = df['InvoiceDate'].dt.month


# In[46]:


monthly_orders = df.groupby('Month').size()


# In[47]:


monthly_returns = df[df['IsReturn']].groupby('Month').size()


# In[49]:


monthly_return_rate = (monthly_returns / monthly_orders) *100
monthly_return_rate = monthly_return_rate.fillna(0).sort_index()

print(monthly_return_rate)


# In[53]:


plt.figure(figsize=(12,6))
bars = plt.bar(monthly_return_rate.index, monthly_return_rate.values, color='skyblue', edgecolor='black')

# Add value labels on top
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10)

plt.ylabel('Return Rate (%)')
plt.xlabel('Month')
plt.title('Return Rate by Month')
plt.xticks(monthly_return_rate.index)  # keeps month numbers 1-12
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# # Do repeat customers return more/less than new customers?

# In[57]:


customer_order_counts = df.groupby('CustomerID').size()


# In[58]:


repeated_customers = customer_order_counts[customer_order_counts > 1].index


# In[59]:


df['CustomerType'] = df['CustomerID'].apply(lambda x:'Repeat' if x in repeated_customers else 'New') 


# In[61]:


cust_orders = df.groupby('CustomerType').size()
cust_returns = df[df['IsReturn']].groupby('CustomerType').size()

cust_return_rate = (cust_returns / cust_orders) * 100
cust_return_rate = cust_return_rate.fillna(0)

print(cust_return_rate)


# Analysis shows that repeat customers have a return rate of 2.2%,
# compared to just 0.29% for new customers — nearly 7.5x higher.
# This insight suggests that while repeat customers drive sales,
# they also drive a disproportionate share of returns,
# highlighting a need for potential policy review or loyalty segmentation.

# In[62]:


cust_return_rate.plot(kind='bar', color=['skyblue','salmon'], edgecolor='black')
plt.ylabel('Return Rate (%)')
plt.title('Return Rate: New vs Repeat Customers')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




