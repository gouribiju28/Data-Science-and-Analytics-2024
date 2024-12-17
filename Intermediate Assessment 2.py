#!/usr/bin/env python
# coding: utf-8

# In[254]:


import pandas as pd
import numpy as np


# In[255]:


df = pd.read_csv("C:\\Users\\Gouri\\Downloads\\train_ctrUa4K.csv")
df


# In[256]:


df.info()


# In[257]:


df.describe()


# In[258]:


num = df.select_dtypes(include="number")
num


# In[259]:


categorical = df.drop(["ApplicantIncome","CoapplicantIncome", "LoanAmount","Loan_Amount_Term","Credit_History"], axis=1)
categorical


# ## PRE PROCESSING

# In[261]:


df.isna().sum()


# In[262]:


num.isna().sum()


# In[310]:


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[312]:


df


# In[314]:


categorical.isna().sum()


# In[316]:


for col in df:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isna().sum()


# In[318]:


df.info()


# In[320]:


df.duplicated().sum()


# In[ ]:





# ## Removing Outliers

# In[270]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(df)
plt.xticks(rotation=45, ha='right')


# In[271]:


import matplotlib.pyplot as plt
def remove_outliers(num, column_name):
    q1 = num[column_name].quantile(0.25)
    q3 = num[column_name].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    num[column_name] = num[column_name].clip(upper=upper_bound)
    num[column_name] = num[column_name].clip(lower=lower_bound)
    return num[column_name]

for col in num:
  num[col] = remove_outliers(num, col)

sns.boxplot(num)
plt.xticks(rotation=45, ha='right')
plt.show()


# ## Scaling

# In[273]:


for c in num:
  plt.hist(num[c])
  plt.title("Histogram of {} column".format(c))
  plt.xlabel(c)
  plt.ylabel("count")
  plt.show()


# In[274]:


for c in df:
  plt.hist(df[c])
  plt.title("Histogram of {} column".format(c))
  plt.xlabel(c)
  plt.ylabel("count")
  plt.show()


# In[275]:


print(df.columns)


# In[276]:


onehot = pd.get_dummies(df, columns=['Gender','Married','Self_Employed','Education'], dtype=int, drop_first=True)
onehot


# In[277]:


df=onehot
df=df.replace(to_replace='3+',value=4)
df


# In[278]:


from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df['Property_Area'] = label_enc.fit_transform(df['Property_Area'])
df


# In[279]:


label_enc = LabelEncoder()
df['Loan_Status'] = label_enc.fit_transform(df['Loan_Status'])
df


# In[280]:


label_enc = LabelEncoder()
df['Loan_Amount_Term'] = label_enc.fit_transform(df['Loan_Amount_Term'])
df


# In[281]:


df['Loan_Amount_Term'].unique()


# In[282]:


from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
scaled_data = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_data,
                         columns=num_cols)
scaled_df.head()


# In[283]:


for c in scaled_df:
  plt.hist(scaled_df[c])
  plt.title("Histogram of {} column".format(c))
  plt.xlabel(c)
  plt.ylabel("count")
  plt.show()


# In[284]:


from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
scaled_df= scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_df,columns=[num_cols])
df[num_cols]=scaled_df
df


# In[285]:


train = df
train


# In[322]:


test = pd.read_csv("C:\\Users\\Gouri\\Downloads\\test_lAUu6dG.csv")
test


# In[324]:


test.info()


# In[326]:


test.isna().sum()


# In[328]:


test_num = test.select_dtypes(include="number")
test_num


# In[330]:


categ_test = test.drop(["ApplicantIncome","CoapplicantIncome", "LoanAmount","Loan_Amount_Term","Credit_History"], axis=1)
categ_test


# In[332]:


test_num.isna().sum()


# In[334]:


test_num['LoanAmount'] = test_num['LoanAmount'].fillna(test_num['LoanAmount'].mean())
test_num['Loan_Amount_Term'] = test_num['Loan_Amount_Term'].fillna(test_num['Loan_Amount_Term'].mean())
test_num['Credit_History'] = test_num['Credit_History'].fillna(test_num['Credit_History'].mean())


# In[336]:


test_num.isna().sum()


# In[338]:


for col in categ_test:
    categ_test[col] = categ_test[col].fillna(categ_test[col].mode()[0])

categ_test.isna().sum()


# In[340]:


test = pd.concat([test_num, categ_test], axis=1)
test


# In[342]:


test.isna().sum()


# In[344]:


test.duplicated().sum()


# In[346]:


sns.boxplot(test)


# In[347]:


import matplotlib.pyplot as plt
def remove_outliers(test_num, column_name):
    q1 = test_num[column_name].quantile(0.25)
    q3 = test_num[column_name].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    test_num[column_name] = test_num[column_name].clip(upper=upper_bound)
    test_num[column_name] = test_num[column_name].clip(lower=lower_bound)
    return test_num[column_name]

for col in test_num:
  test_num[col] = remove_outliers(test_num, col)

sns.boxplot(test_num)
plt.xticks(rotation=45, ha='right')
plt.show()


# In[349]:


for c in test:
  plt.hist(test[c])
  plt.title("Histogram of {} column".format(c))
  plt.xlabel(c)
  plt.ylabel("count")
  plt.show()


# In[351]:


onehot_test = pd.get_dummies(test, columns=['Gender','Married','Self_Employed','Education'], dtype=int, drop_first=True)
onehot_test


# In[352]:


test=onehot_test
test=test.replace(to_replace='3+',value=4)
test


# In[353]:


from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
test['Property_Area'] = label_enc.fit_transform(test['Property_Area'])
test


# In[304]:


label_enc = LabelEncoder()
test['Loan_Status'] = label_enc.fit_transform(test['Loan_Status'])
test


# In[305]:


label_enc = LabelEncoder()
test['Loan_Amount_Term'] = label_enc.fit_transform(test['Loan_Amount_Term'])
test


# In[306]:


# scaler = StandardScaler()
# test_num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
# test_scaled_data = scaler.fit_transform(test[test_num_cols])
# test_scaled_df = pd.DataFrame(test_scaled_data,
#                          columns=test_num_cols)
# test_scaled_df.head()

# scaler = StandardScaler()
# num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
# scaled_data = scaler.fit_transform(df[num_cols])
# scaled_df = pd.DataFrame(scaled_data,
#                          columns=num_cols)
# scaled_df.head()

scaler = StandardScaler()
test_num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
scaled_test= scaler.fit_transform(test[test_num_cols])
scaled_test = pd.DataFrame(scaled_test,columns=[test_num_cols])
test[test_num_cols]=scaled_test
test


# In[307]:


for c in scaled_test:
  plt.hist(scaled_test[c])
  plt.title("Histogram of {} column".format(c))
  plt.xlabel(c)
  plt.ylabel("count")
  plt.show()


# In[450]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

#training dataset
x = train.drop(['Loan_Status','Loan_ID'],axis=1)
y = train['Loan_Status']

# test dataset
x_test1 = test.drop(['Loan_ID'], axis=1)
# y_test1 = test['Loan_Status']

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=52)


# ### Logistic Regression Model

# In[489]:


model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[491]:


accuracy_score(y_test, y_pred)


# In[493]:


accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="weighted")
recall=recall_score(y_test,y_pred,average="weighted")
f1=f1_score(y_test,y_pred,average="weighted")
print("Accuracy is",accuracy,"\nPrecison is",precision,"\nRecall is",recall,"\nF1 score is",f1)


# ### kNN

# In[496]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y1_pred = knn.predict(x_test)


# In[498]:


accuracy = accuracy_score(y_test, y1_pred)
print(f"Accuracy: {accuracy}")


# In[500]:


test_accuracy = accuracy_score(y_test, y1_pred)
test_precision = precision_score(y_test, y1_pred, average='weighted')
test_recall = recall_score(y_test, y1_pred, average='weighted')
test_f1 = f1_score(y_test, y1_pred, average='weighted')
print("Accuracy is",test_accuracy,"\nPrecison is",test_precision,"\nRecall is",test_recall,"\nF1 score is",test_f1)


# In[434]:


example_point = np.array([[0, 0.07, -0.55, 0, 9, 1, 2,1,1,0,1]])
predicted_class = knn.predict(example_point)
print(f"Predicted class for example point: {predicted_class}")


# In[436]:


ex = np.array([[1,1.2,-0.63,1,8,0,1,0,0,1,0]])
predicted_class = knn.predict(ex)
print(f"Predicted class for example point is: {predicted_class}" )


# ### Naive Bayes

# In[502]:


nb = GaussianNB()
nb.fit(x_train, y_train)
ypred = nb.predict(x_test)


# In[508]:


accuracy=accuracy_score(y_test,ypred)
accuracy


# In[506]:


testaccuracy = accuracy_score(y_test, ypred)
testprecision = precision_score(y_test, ypred, average='weighted')
testrecall = recall_score(y_test, ypred, average='weighted')
testf1 = f1_score(y_test, ypred, average='weighted')
print("Accuracy is",testaccuracy,"\nPrecison is",testprecision,"\nRecall is",testrecall,"\nF1 score is",testf1)


# ### Random Forest

# In[510]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[511]:


accuracy=accuracy_score(y_test,y_pred)
accuracy


# ## Since logistic regression gives higher accuracy among the four models tested, we choose logstic regression model for testing the test data set

# In[457]:


X_train=x
Y_train=y
X_test=x_test1


# In[465]:


print(X_test.columns.tolist())
print(X_train.columns.tolist())


# In[467]:


X_test=X_test[X_train.columns]
print(X_test.columns.tolist())
print(X_train.columns.tolist())


# In[469]:


model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)


# In[477]:


Y_pred.shape


# In[554]:


sample = pd.read_csv("C:\\Users\\Gouri\\Downloads\\sample_submission_49d68Cx.csv")
sample.head()
dtest = pd.read_csv("C:\\Users\\Gouri\\Downloads\\test_lAUu6dG.csv")


# In[483]:


Y_pred


# In[576]:


sample['Loan_ID']= dtest['Loan_ID']
sample['Loan_Status']=Y_pred


# In[578]:


sample['Loan_Status'].replace(0, 'N', inplace=True)
sample['Loan_Status'].replace(1, 'Y', inplace=True)
sample


# In[570]:


sample.to_csv("/Users/Gouri/Downloads/Submission-IA2.csv",index=False)


# In[572]:


submission.shape


# In[574]:


sample.shape


# In[ ]:




