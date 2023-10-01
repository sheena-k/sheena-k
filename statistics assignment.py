#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT ON STATISTICS

# In[45]:


import numpy as np
import pandas as pd


# # 1) Read data

# In[4]:


df = pd.read_csv("Students_Performance.csv")


# In[5]:


df.head()


# # 1a) Males and Females participated in the test

# In[6]:


gender = df["gender"].value_counts()


# In[7]:


gender


# # 1b) Students' Parental level of Education

# In[33]:


df["parental level of education"].value_counts()


# # 1 c (i)average for math, reading and writing based on Gender

# In[18]:


Avg_gender= df[["gender","math score","reading score","writing score"]].groupby("gender").mean()


# In[19]:


Avg_gender


# # 1 c (ii)average for math, reading and writing based on test preparation course

# In[20]:


Avg_test=df[["test preparation course","math score","reading score","writing score"]].groupby("test preparation course").mean()


# In[21]:


Avg_test


# # 1d(i) the scoring variation for math, reading and writing based on gender

# In[22]:


Var_gen=df[["gender","math score","reading score","writing score"]].groupby("gender").std()


# In[23]:


Var_gen


# # 1d(ii) the scoring variation for math, reading and writing based on test

# In[25]:


Var_test=df[["test preparation course","math score","reading score","writing score"]].groupby("test preparation course").std()


# In[26]:


Var_test


#  # 1(e) Top 25% of students based on their math score

# In[27]:


df.nlargest(250,["math score"])


# # Case Study on Testing of Hypothesis

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


# # 2) Sales Data

# In[8]:


sales_data = pd.read_csv("Sales_add.csv")


# # Reading data

# In[9]:


sales_data.head()


# # Null value Analysis

# In[40]:


sales_data.info()


# In[42]:


sales_data.isnull().sum()


# # Descriptive Analysis

# In[43]:


sales_data.describe()


# **from the above analysis it is conculded that there is no null values in dataset and it has 22 row and 5 columns

# # PERFORMING CASE STUDY
# 
# checking for outliers in the data

# In[15]:


fig, (ax0,ax1) = plt.subplots(figsize = (12,8), nrows=1, ncols=2)

sns.boxplot( y = "Sales_before_digital_add(in $)", data =sales_data, ax = ax0)
ax0.set(title = "Sales_before_digital_add(in $)")
sns.boxplot( y = "Sales_After_digital_add(in $)", data =sales_data, ax = ax1)
ax1.set(title = "Sales_After_digital_add(in $)")


# ** from the above, no outliers were present.

# # 2 a)CASE1
# For finding there is any increase in sales after stepping into digital marketing.
# We will be performing this study in 4 steps:
# 
# STEP 1
# Define the Null and Alternate Hypothesis and set the Significance level.
# 
# Null Hypothesis:
# Ho : Sales after digital advertising will be less than or equal to the sales before digital advertising.
# 
# Alternate Hypothesis:
# HA : Sales after digital advertising will be greater than the sales before digital advertising.
# 
# The Confidence level for this test = 95% and the level of Significance,alpha = 0.05.

# STEP 2
# calculate t-score and p-value

# In[43]:


sales_before = sales_data[["Sales_before_digital_add(in $)"]]

sales_after = sales_data[["Sales_After_digital_add(in $)"]]

#conducting a 1 tail t TEST at alpha =0.05 and t-critical = 1.721 with dof =21( degree of freedom = n-1, 22-1 =21)

t_score, p = stats.ttest_rel(sales_after,sales_before, alternative="greater")
print("The Test statistic scores are \n t-score:", t_score)
print("p value",p)


# STEP 3
# 
# Calculate t-score with the critical value of t at 0.05 level of significance(t_critical = 1.721).

# In[32]:


t_critical = 1.721
if t_score > t_critical:
    print("\nReject the Null Hypothesis\n\n")
elif t_score <= t_critical:
    print("\nDon't reject the Null Hypothesis\n")


# STEP 4
# #From the above Testing we can reach to the following about our Hypothesis:
# 
# As the calculated t-score > critical t-score value (significance level at 5% or 0.05),REJECT NULL HYPOTHESIS.
# We can say that there is a significant increase in sales after doing Digital advertisements.
# 

# # 2 b) CASE 2
# 
# Checking whether there is any dependency between the features “Region” and “Manager”
# Same as the previous case we'll follow a similar procedure.
# STEP 1
# 
# Define the Null and Alternate Hypothesis and set the Significance level.
# 
# Null Hypothesis:
# Ho : There is NO significant dependency between the Region and the Manager features.
# 
# Alternate Hypothesis:
# HA : There is a significant amount of dependency between the Region and the Manager features.
# 
# The Confidence level for this test will be 95% & set the level of Significance as alpha = 0.05.

# STEP 2

# In[31]:


# Extracting the Required Features, performing a crosstab on them and assigning it to a new variable
data_crosstab = pd.crosstab(sales_data["Region"],sales_data["Manager"])
data_crosstab


# STEP 3

# In[44]:


stat, p, dof, expected = chi2_contingency(data_crosstab)

print("The Test chi-square value is :",stat)
print("\nThe p-Value is :",p )
print("\nThe Degree of freedom is :",dof)

chi2_critical = 9.488 # the chi2 value at alpha = 0.05 and dof = 4

if stat > chi2_critical:
    print("Reject the Null Hypothesis")
elif stat < chi2_critical:
    print ("Do not Reject the Null Hypothesis")


# STEP 4

# From the above Testing we can reach to following about the Hypothesis:
# 
# The calculated chi2 value < Critical chi2 value at 0.05 significance level and the calculated p-value > 0.05.
# Hence,we can conclude that there is no significant relationship between the features Regions and Managers.
# 
# **Overall Conclusion from the Study conducted can be Summerized as
# 1)There is significant amount of increase in sales generated after the company started investing in Digital Marketing.
# 2)There is not significant dependency between the Regions and the Managers associated with the regions.
