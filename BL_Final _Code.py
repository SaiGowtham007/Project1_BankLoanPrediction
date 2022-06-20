#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install --upgrade pandas


# In[2]:


#pip install plotting


# In[3]:


#pip install ggplot


# In[4]:


#pip install fancyimpute


# In[5]:



import pandas as pd
bank=pd.read_csv('C:/Users/saigo/Desktop/s/bank-loan.csv')
print(bank.head(5))
bank.head(5).describe()


# In[6]:


bank.rename(columns = {'default':'result'}, inplace = True)


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from fancyimpute import KNN  


# In[8]:


columns=list (bank.columns)
columns


# In[9]:


bank.tail(5)


# In[10]:


bank.info()


# In[11]:


bank.shape


# In[12]:


bank.describe()


# In[13]:


############ univariate analysis and bivariate analysis ##########################

#Analysis for single variable in the dataset is Univariate Analysis and plotting relation between 2 variables is bivariate analysis.
sns.set(rc = {'figure.figsize':(15,8)})
sns.countplot(bank["age"])
# age_20 = bank[bank["age"] == 20]
# print()


# In[14]:



# it means there are missing values in var

bank['result'].unique()


# In[15]:


bank['result'].value_counts()


# In[16]:


(850-517-183) # nan values


# In[17]:


columns


# In[18]:


# replacing nan values with 0 in each and every column
for i in columns:
  bank[i] = bank[i].fillna(0)


# In[19]:


columns[0]


# In[20]:


sns.countplot( x = "result" , data = bank,)


# In[21]:


sns.countplot( x = "ed" , data = bank, hue = "result")


# 
# ## Outliers Analysis and Synthesis
# 

# In[22]:


print(bank.head())
bank.describe()


# In[23]:


#scatter plot for outlier Analysis
for i in columns:
  if i!='result':
    plt.scatter(bank['result'],bank[i])
    plt.title('Outlier Analysis')
    plt.xlabel('Count')
    plt.ylabel(i)
    plt.show()


# In[24]:


bank.info()


# # Now detect and replace Outliers

# In[25]:



get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(bank['income'])  


# In[26]:


# 2.Detect outliers and replace  NAn later impute by KNN imputation

#Extract quartiles
q75, q25 = np.percentile(bank['income'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
bank.loc[bank['income']  < minimum,:'income'] = np.nan

bank.loc[bank['income'] > maximum,:'income'] = np.nan

#Calculate missing value
bank.income.isnull().sum()

# missing_val = pd.DataFrame(bank.isnull().sum())


# In[27]:


#replacing NaNs with Knn imputation
bank = pd.DataFrame(KNN(k = 3).fit_transform(bank), columns = bank.columns)


# In[28]:


bank.result.isnull().sum()


# In[29]:


## Now check outliers got imputed or not 
get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(bank['income']) 


# As of now we got data having zero missing values and Outliers

# # Next step = Feature Selection

# #### Selection of categorical vars -- Chi_Sqr Test of Independance
# 
# #### Selection of Numerical vars i.e. ( cnames ) --  Correlation analysis 
# 

# In[30]:


cnames=['age',	'employ'	,'address',	'income'	,'debtinc'	,'creddebt',	'othdebt']


# In[31]:


df_corr = bank.loc[:,cnames]
df_corr


# In[32]:


#Set the width and hieght of the correlation plot

f, ax = plt.subplots(figsize = (7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax , annot = True )


# In[33]:



df_1 = bank.corr()
sns.heatmap(df_1 , annot = True , cmap = "coolwarm")


# #### As u can see above in plot, no any variable is identical to other var, it means these vars are no hiighly correlated variables,  so we have to carry all variables and we will put all of them in model developement as all vars are imp right now.

# In[34]:


bank['ed'].unique()


# In[35]:


bank['result'].unique()


# In[36]:


bank['ed'] = pd.Categorical(bank['ed'])
print(bank.ed.dtype)

bank['result'] = pd.Categorical(bank['result'])
print(bank.result.dtype)


# As we can see here both ed and result columns are categorical
# so we perform chi 2 statistics to see the relation between them

# In[37]:


cat_names = ["ed"]
from scipy.stats import chi2_contingency  # for chi-sqr test and comtingency table
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(bank['result'], bank[i]))
    print(p)


# As we can, see , p Value of cat var (ed) is > 0.05 , It means We Accept Null Hhypothesis saying that these two variables are , not imp to each other, n we can drop any one of them instead of carrying both same vars.

# In[38]:


#  Now remove less important features  / Diamension reduction
from copy import deepcopy
bank = bank.drop(['ed'], axis=1)
bank.head(2)
bank_1=deepcopy(bank)


# In[39]:


bank_1.head()


# # Feature Scaling 
# ## scale tht imp features in measurable units
# ### ___1) Scaling by Normalization
# ### __or_2) Scaling by Standardization

# In[40]:


#check Normality by Histogram Before Normalization / Standerdization

get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(bank['age'], bins='auto')


# In[41]:


# Again verify it

get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(bank['income'], bins='auto')


#  Since we can see that , data is not normallaly distributed , Hence go for **Normalization** 1st instead of Stdn

# In[42]:


cnames


# In[43]:


#Nomalisation

for i in cnames:
    print(i)
    bank[i] = (bank[i] - min(bank[i]))/(max(bank[i]) - min(bank[i]))


# # Machine Learning Algorithms
# 
# 
# 

# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


bank['result'] = bank['result'].astype('int64')#chaning dtype for result from float to int


# In[47]:


bank.head(2)
print(bank.age)


# In[48]:


#Now divide the data into train and test

X= bank.values[:,0:7]     #saving all   var's in X
Y= bank.values[:,7]        #saving 1 dep var in Y


# In[49]:


pd.DataFrame(X).head(2)


# In[50]:


#Now split the data into train and test
   #devided 80% and 20% of ALL var's obs (except 'default' var) in X_train and into X_test Respectively
   #devided 80% and 20% of Dep.Var's obs ( default var's) into y_train and into y_test Respectively

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=12)


# In[51]:


import warnings
warnings.filterwarnings("ignore")


# In[52]:


models=[]


from sklearn.linear_model import LogisticRegression
logre = LogisticRegression()
models.append(logre)

from sklearn.svm._classes import LinearSVC
lsvc=LinearSVC()
models.append(lsvc)

from sklearn.linear_model import LogisticRegressionCV
logrecv=LogisticRegressionCV()
models.append(logrecv)

from sklearn.svm import SVC
svc=SVC()
models.append(svc)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
models.append(sgd)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
models.append(nb)

from sklearn.tree import  DecisionTreeClassifier
dt=DecisionTreeClassifier()
models.append(dt)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
models.append(rf)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
models.append(knn)


# In[53]:


from sklearn.model_selection import KFold,cross_val_score
cv=KFold(10)
mls=[]
scores=[]

for model in models:
  score=cross_val_score(model,X_train,y_train,cv=cv)
  print(str(model).split('(')[0]+' :- ',np.mean(score))
  mls.append(str(model).split('(')[0])
  scores.append(int(np.mean(score)*100))


# In[54]:


import matplotlib.pyplot as plt
ml_scores= pd.DataFrame({"models": mls,"scores": scores})
import seaborn as sns
ml_scores.sort_values('scores')

plt.figure(figsize=(25,10))
sns.barplot(x='models',y="scores",data=ml_scores,palette="Blues_d",order=ml_scores.sort_values('scores').models)


def addtext(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],str(y[i])+'%')
addtext(models,sorted(scores)) 
plt.xlabel("models",size=16)
plt.ylabel("scores", size=16)
plt.show()


# In[55]:



for model in models:
  model.fit(X_train,y_train)
  print(model)
  print('Train Accuracy:-',accuracy_score(y_train,model.predict(X_train)))
  print('Test Accuracy:-',accuracy_score(y_test,model.predict(X_test)))
  print()


# among all these alogirthms we choose Logistic Regression because of its train and test scores which is best fit

# In[56]:


X_train[0]


# In[57]:


cnames


# In[59]:


import pickle
logre_model=pickle.load(open('C:/Users/saigo/Desktop/s/logre_model','rb'))


# In[60]:


X_train[0]


# In[61]:


user_input=np.array(X_train[0])


# In[62]:


user_input=user_input.reshape(1,-1)


# In[63]:


logre_model.predict(user_input)[0]


# In[ ]:


y_train[0]


# In[ ]:


pip install pywebio


# In[ ]:


#import pickle
#saving ml model
#pickle.dump(logre,open('C:/Users/saigo/Desktop/s/logre_model','wb'))
#loading ml model
#logre_model=pickle.load(open('C:/Users/saigo/Desktop/s/logre_model','rb'))


# In[ ]:


app=Flask(__name__)
logre_model=pickle.load(open('C:/Users/saigo/Desktop/s/logre_model','rb'))
#sample inputs
#[0.11111111, 0.1       , 0.16129032, 0.02247191, 0.27184466,0.00297762, 0.04484908] - y=1
def predict(): 
    user_input=[]
    ss_user_input=[]
    for i in cnames:
        x=input(i+' :- ',type=FLOAT)
        user_input.append(x)
        
    for i in range(len(user_input)):
        ss = (user_input[i] - min(user_input))/(max(user_input) - min(user_input))
        ss_user_input.append(ss)

    ss_user_input=np.array(ss_user_input)
    ss_user_input=ss_user_input.reshape(1,-1)
    
    prediction=logre_model.predict(ss_user_input)[0]
    if prediction==1.0:
        prediction='Yes'
    else:
        prediction='No'
    put_text('prediction = %r' % prediction)
        


app.add_url_rule('/','webio_view',webio_view(predict),methods=['GET','POST','OPTIONS'])
app.run(host='localhost',port=88)


# In[ ]:


X_train[1]


# In[ ]:


y_train[0]


# In[ ]:


sampleList = np.array(bank_1[:2])
sampleList


# In[ ]:


bank_1.head()


# In[ ]:


#[32.35619303,  7.23615625,  2.60150296, 89.80156532,  9.3       ,11.359392  ,  5.008608  ,       ] - ouput -1
#[27.        , 10.        ,  6.        , 31.        , 17.3       ,1.362202  ,  4.000798  ,      ]  -output -0 

