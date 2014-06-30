'''
Created on Jun 29, 2014

@author: domingolara
'''


import numpy as np
import pandas as pd

def recodeVariables(df):
    # Map gender column Female = > 0 and Male => 1
    df['Gender'] = df['Sex'].map( {'female':0 ,'male':1} ).astype(int)
    df['Embarked_class'] = df['Embarked'].map( {'C':0, 'Q':1, 'S':2})
    return df


def fixMissingAgeValues(df):
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass']==j+1)]['Age'].dropna().median()
            
    df['AgeFill'] = df['Age']
    
    for i in range(0,2):
        for j in range(0,3) :
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    return df



if __name__ == '__main__':
    pass