
# coding: utf-8

# ## Demo script for Market Bucket Analysis or Association Rules Mining

# Overview:
# 
# •	Association rule mining/market bucket analysis popular method to identify relationship between 
#     different variables in a dataset. It is also called as if/then analysis
#     If a person buys onions and potatoes (Antecedent) together, the he is likely to buy hamburger bread (Consequent)
# 
# Why Association Rules Mining: 
# 
# •	Nowadays online stores like Amazon.com, Alibaba.com are getting very popular amongst the customers, 
#     which has increased competition for the retail stores and have left very market space. 
#     So, to compete with the online stores, retail stores can utilizes insights from association rule mining
#     /market bucket analysis to come up with better marketing/advertising strategies that target customer on personalized basis 
#     (providing them offers on product(s) they frequently buy), 
#     thus increasing the chances of the customer to come to the store and make a sale.

# To keep the examples simple and easy to understand I am using dummy data.
# Dummy transaction data:
#     transactions = [
#     ['beer', 'nuts'],
#     ['beer', 'cheese','potato'],
#     ['beer', 'nuts'], 
#     ['beer', 'nuts','cola'],
#     ['cola','soap'],
#     ['cola','soap','shampoo'],
#     ['cola','soap','nuts']
#     ]

# Support: Percentage of orders that contains the item set. In the example (beer, nuts) are present in 3 out of 7 order, 
#          therefore: 
#          support{beer,nuts} = 3/7 or 42.85%
#          
#          In real world scenario we mights have 100,000 of transactions, there for we set very small value for the support 
# 
# Confidence:   Percentage of times nuts is purchased given that beer is purchased. 
#               Formula: confidence{A->B} = support{A,B} / support{A}  
#               In the example, support{beer,nuts} = 3/7 and support{beer} = 4/7, therefore:
#               confidence{beer->nuts} = support{beer,nuts}/support{beer} = 3/4 = 0.75
#               
#               A confidence value of 0.75 implies that out of all orders that contain beer, 75% of them also contain nuts.
#               
#               It is important to note that in confidence{beer->nuts} != confidence{nuts->beer}
# 
# Lift:   Indicates if there is a relationship between two products or they occur together by chance or randomly
#         Lift{beer,nuts} = support{beer,nuts}/support{beer}*support{nuts}
#         
#         Looking at the formula above, lift of 1 indicates that the likelyhood of beer and nuts occoring together is same as beer
#         and nuts occoring separately, indicating that there is not realtionship between beer and nuts.
#         
#         To summarize:
#         •	lift = 1 implies no relationship between beer and nuts 
#             (ie: beer and nuts occur together only by chance or randomly)
#             
#         •	lift > 1 implies that there is a positive relationship between beer and nuts
#             (ie:  beer and nuts occur together more often than by chance/random)
#             
#         •	lift < 1 implies that there is a negative relationship between beer and nuts
#             (ie:  beer and nuts occur together less often than by chance/random)
#     

# In[47]:


# Link for apyori package: https://pypi.python.org/pypi/apyori/1.1.1
from apyori import apriori
from apyori import dump_as_json
import json
import io
import pandas as pd


# In[104]:


transactions = [
    ['beer', 'nuts', 'pen'],
    ['beer', 'cheese','potato'],
    ['beer', 'nuts'], 
    ['beer', 'nuts','cola'],
    ['cola','soap'],
    ['cola','soap','shampoo'],
    ['cola','soap','nuts']
]
transactions


# #### Get the frequency of the products in the transaction.

# In[143]:


import collections, itertools

freq_of_item = collections.defaultdict(int)  # 0 by default
for x in itertools.chain.from_iterable(transactions):
    freq_of_item[x] += 1
    
print(freq_of_item)


# In[117]:


#We can tweak the inputs to the algoritm to get the desired results in output
results = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 2)


# In[118]:


results_list = list(results)

results_list


# #### As the output of the algorithm is difficult to understand and present, I have written a parser to create a dataframe of the results

# In[119]:


summary_df = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence','Lift'))

Support =[]
Confidence = []
Lift = []
Items = []
Antecedent = []
Consequent=[]

for RelationRecord in results_list: 
    for ordered_stat in RelationRecord.ordered_statistics:
        Support.append(RelationRecord.support)
        Items.append(RelationRecord.items)
        Antecedent.append(ordered_stat.items_base)
        Consequent.append(ordered_stat.items_add)
        Confidence.append(ordered_stat.confidence)
        Lift.append(ordered_stat.lift)

summary_df['Items'] = Items                                   
summary_df['Antecedent'] = Antecedent
summary_df['Consequent'] = Consequent
summary_df['Support'] = Support
summary_df['Confidence'] = Confidence
summary_df['Lift']= Lift


# In[ ]:


summary_df.groupby('Antecedent')


# In[120]:


summary_df.sort_values('Lift', ascending = False)


# There is +ive relationship between cheese & potato, a confidence value of 1.00 implies that out of all 
# orders that contain cheese, 100% of them also contain potato.
# 
# Therefore according to the data, cheese and potato should be kept side by side on the shelf to increase sales.
