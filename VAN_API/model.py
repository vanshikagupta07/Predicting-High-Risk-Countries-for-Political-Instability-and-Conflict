import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv('Master_refined.csv')

#Replacing 'Czechia' with 'Czech Republic' as both are same countries
df['Country'] = df['Country'].replace('Czechia', 'Czech Republic')

#Replacing 'Eswatini' with 'Swaziland' as both are same countries
df['Country'] = df['Country'].replace('Eswatini', 'Swaziland')

#Since Rank is a categorical variable so we need to extract the numerical part
df['Rank_numerical'] = df.Rank.str.extract('(\d+)') # captures numerical part
df['Rank_categorical'] = df['Rank'].str[-2:] # captures the first letter

#Dropping 'Rank' and 'Rank_categorical' as we extracted the numerical part of 'Rank' is in 'Rank_numerical'
df=df.drop(columns=['Rank','Rank_categorical'])

#Changing the dtype of 'Rank_numerical'
df['Rank_numerical']=df['Rank_numerical'].astype(str).astype(int) 

#encoding 'Country' as it is a categorical variable
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df['Country']= label_encoder.fit_transform(df['Country']) 

X = df.iloc[:, :15]
y = df.iloc[:, -1]

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dregressor = DecisionTreeRegressor()
dregressor.fit(X.drop(['C1: Security Apparatus',
 'C2: Factionalized Elites',
 'C3: Group Grievance',
 'E1: Economy',
 'E2: Economic Inequality',
 'E3: Human Flight and Brain Drain',
 'P1: State Legitimacy',
 'P2: Public Services',
 'P3: Human Rights',
 'S1: Demographic Pressures',
 'S2: Refugees and IDPs',
 'X1: External Intervention'],axis=1), y)

import pickle

# Saving model to disk
pickle.dump(dregressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
