#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DS Basics
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt; plt.rcdefaults()
import statistics


# In[2]:


#Import data
URL_TRAIN='https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/jump2digital/dataset/train.csv'
URL_TEST='https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/jump2digital/dataset/test_X.csv'
train=pd.read_csv(URL_TRAIN)
data_test=pd.read_csv(URL_TEST,sep=";")
#train=pd.read_csv('train.csv')
#data_test=pd.read_csv('test_X.csv',sep=';')


# # About data
# order_id: Número de identificación del pedido.
# 
# local_time: Hora local a la que se realiza el pedido.
# 
# country_code: Código del pais en el que se realiza el pedido.
# 
# store_address: Número de tienda en a la que se realiza el pedido.
# 
# payment_status: Estado del pedido.
# 
# n_of_products: Número de productos que se han comprado en ese pedido.
# 
# products_total: Cantidad en Euros que el usuario ha comprado en la app.
# 
# final_status: Estado final del pedido (este será la variable 'target' a predecir) que indicara si el pedido será finalmente entregado o cancelado. Hay dos tipos de estado:

# ##  ¿Cuáles son los 3 paises en los que más pedidos se realizan?
# 

# In[3]:


train['country_code'].unique()


# In[4]:


country_n_delivered=train.groupby('country_code').size()
country_n_delivered_ord=country_n_delivered.sort_values(ascending=False)
top3_country_deliver=country_n_delivered_ord[0:3]
top3_country_deliver


# In[5]:


print('Los 3 paises con mas pedidos son: ', top3_country_deliver)


# ## ¿Cuáles son las horas en las que se realizan más pedidos en España?
# 

# In[6]:


spain=train[train['country_code']=='ES']
spain_time=spain['local_time'].to_frame()
spain_hour = spain_time['local_time'].str[:2].to_frame()
spain_hour_size= spain_hour.groupby('local_time').size().sort_values(ascending=False).to_frame()
top3_spain_hour=spain_hour_size[:3]
print('Las horas que españa tiene mas pedidos son: las',
      top3_spain_hour.index[0],',las',top3_spain_hour.index[1],
      ' y las',top3_spain_hour.index[2])


# ## ¿Cuál es el precio medio por pedido en la tienda con ID 12513?
# 

# In[7]:


store_12513=train[train['store_address']==12513]
mean_store_12513=str(round(statistics.mean(store_12513['products_total']),2))
print('El precio medio por pedido de la tienda 12513 es de: ',mean_store_12513)


# ##  Teniendo en cuenta los picos de demanda en España, si los repartidores trabajan en turnos de 8horas.
# Turno 1 (00:00-08:00)
# Turno 2 (08:00-16:00)
# Turno 3 (16:00-00:00)
# 
# Qué porcentaje de repartidores pondrías por cada turno para que sean capaces de hacer frente a los picos de demanda. (ej: Turno 1 el 30%, Turno 2 el 10% y Turno 3 el 60%).

# In[8]:


train_ES=train[train['country_code']=='ES']
len(train_ES)


# In[9]:


train_ES['hora']=train_ES['local_time'].str[:2].to_frame()
horas_ord=sorted(train_ES['hora'])
horas_ord=[int(horas_ord_num) for horas_ord_num in horas_ord]


# In[10]:


turno_1=[]
turno_2=[]
turno_3=[]
for i in horas_ord:
    if i in list(range(0,8)) :
        turno_1.append(i)
    if i in list(range(8,16)):
        turno_2.append(i)
    if i in list(range(16,24)):
        turno_3.append(i)


# In[11]:


por_turno_1=round(len(turno_1)/len(horas_ord)*100,2)
por_turno_2=round(len(turno_2)/len(horas_ord)*100,2)
por_turno_3=round(len(turno_3)/len(horas_ord)*100,2)
print('Para el primer turno del día, debería haber un',por_turno_1,'%, un',por_turno_2,'% para el segundo \ny para el último turno un',por_turno_3)


# ## Realiza un modelo predictivo de machine learning a partir del dataset 'train.csv' en el cual a partir de las variables predictoras que se entregan en el dataset 'test_X' se pueda predecir si el pedido se cancelará o no (columna 'final_status').
# 

# In[12]:


df=train
df


# In[13]:


hora=df['local_time'].str[:2].to_frame()
df['Hora']=df['local_time'].str[:2].to_frame()
minuto_aux=df['local_time'].str[3:5].to_frame()


# In[14]:


minuto_aux['local_time']=minuto_aux['local_time'].astype(int)


# In[15]:


local_timeQ=[]
for i in minuto_aux['local_time']:
    if i in range(0,14):
        i=1
        local_timeQ.append(i)
    elif i in range(15,29):
        i=2
        local_timeQ.append(i)
    elif i in range(30,44):
        i=3
        local_timeQ.append(i)
    else:
        i=4
        local_timeQ.append(i)


# In[16]:


df['TimeQe']=local_timeQ


# In[17]:


df


# In[18]:


l = ['AR', 'GT', 'CR', 'ES', 'PE', 'PA', 'FR', 'IT', 'TR', 'EC', 'RO','KE',
     'UA', 'PT', 'DO', 'GE', 'MA', 'PR', 'CL', 'EG', 'UY', 'CI','BR']
n=list(range(1,24))
df['country_num']=train['country_code']
df['country_num'].replace(l,n,inplace=True)


# In[19]:


df['payment_status'].unique()
df=df.replace('PAID',1)
df=df.replace('NOT_PAID',2)
df=df.replace('DELAYED',0)


# In[20]:


cor_df=df.corr()
cor_df


# In[21]:


print(df[df['final_status']==1].count()[1])
print(df[df['final_status']==0].count()[1])


# In[22]:




y=df['final_status']
x=df.loc[:,[ 'store_address','payment_status', 'n_of_products', 'products_total',
            'Hora', 'TimeQe', 'country_num']]

#Train/Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=142,shuffle=True)


ros = RandomOverSampler(random_state=42,)
variablex_ros, y_ros = ros.fit_resample(X_train, y_train)

#Modeling
tree_classifier= DecisionTreeClassifier()
model=tree_classifier.fit(variablex_ros,y_ros)

#Predictions
preds = model.predict(X_test)

plot_confusion_matrix(model,X_test,y_test,include_values=bool)
#plt.title('Confusion matrix with demographic variable')
#plt.show()
#plt.savefig('Conf_matr_region.amount.png')

print('SUMMARY OF CONFUSION MATRIX WITH ALL VARIABLE AS INDEPENDENT FEATURE')
print(classification_report(y_test, preds))


# In[23]:


y=df['final_status']
x=df.loc[:,[ 'store_address','payment_status','products_total']]
#Train/Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=142,shuffle=True)


ros = RandomOverSampler(random_state=42,)
variablex_ros, y_ros = ros.fit_resample(X_train, y_train)

#Modeling
tree_classifier= DecisionTreeClassifier()
model=tree_classifier.fit(variablex_ros,y_ros)

#Predictions
preds = model.predict(X_test)

plot_confusion_matrix(model,X_test,y_test,include_values=bool)
#plt.title('Confusion matrix with demographic variable')
#plt.show()
#plt.savefig('Conf_matr_region.amount.png')

print('SUMMARY OF CONFUSION MATRIX WITH ALL VARIABLE AS INDEPENDENT FEATURE')
print(classification_report(y_test, preds))


# In[24]:


y=df['final_status']
x=df.loc[:,[ 'store_address','payment_status','products_total','Hora']]
#Train/Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=142,shuffle=True)

ros = RandomOverSampler(random_state=42,)
variablex_ros, y_ros = ros.fit_resample(X_train, y_train)

#Modeling
tree_classifier= DecisionTreeClassifier()
model=tree_classifier.fit(variablex_ros,y_ros)

#Predictions
preds = model.predict(X_test)

plot_confusion_matrix(model,X_test,y_test,include_values=bool)
#plt.title('Confusion matrix with demographic variable')
#plt.show()
#plt.savefig('Conf_matr_region.amount.png')

print('SUMMARY OF CONFUSION MATRIX WITH ALL VARIABLE AS INDEPENDENT FEATURE')
print(classification_report(y_test, preds))


# In[25]:


y=df['final_status']
x=df.loc[:,[ 'store_address','payment_status','products_total','Hora']]
#Train/Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=142,shuffle=True)



#Modeling
tree_classifier= DecisionTreeClassifier()
model=tree_classifier.fit(X_train,y_train)

#Predictions
preds = model.predict(X_test)

plot_confusion_matrix(model,X_test,y_test,include_values=bool)
#plt.title('Confusion matrix with demographic variable')
#plt.show()
#plt.savefig('Conf_matr_region.amount.png')

print('SUMMARY OF CONFUSION MATRIX WITH ALL VARIABLE AS INDEPENDENT FEATURE')
print(classification_report(y_test, preds))


# In[26]:


#preds = model.predict(data_test)


# In[27]:


y=df['final_status']
x=df.loc[:,[ 'store_address','payment_status','products_total','n_of_products']]
#Train/Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=142,shuffle=True)


ros = RandomOverSampler(random_state=42,)
variablex_ros, y_ros = ros.fit_resample(X_train, y_train)

#Modeling
tree_classifier= DecisionTreeClassifier()
model_1=tree_classifier.fit(variablex_ros,y_ros)

#Predictions
preds = model_1.predict(X_test)

plot_confusion_matrix(model,X_test,y_test,include_values=bool)
#plt.title('Confusion matrix with demographic variable')
#plt.show()
#plt.savefig('Conf_matr_region.amount.png')

print('SUMMARY OF CONFUSION MATRIX WITH ALL VARIABLE AS INDEPENDENT FEATURE')
print(classification_report(y_test, preds))


# In[28]:


data_test=data_test.replace('PAID',1)
data_test=data_test.replace('NOT_PAID',2)
data_test=data_test.replace('DELAYED',0)


# In[29]:


preds = model_1.predict(data_test.loc[:,['store_address','payment_status','products_total','n_of_products']])
preds


# In[31]:


np.savetxt("preds.csv", preds,delimiter=',',fmt = '%s')

