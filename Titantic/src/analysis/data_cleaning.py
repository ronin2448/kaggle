'''
Created on Jun 29, 2014

@author: domingolara
'''


import numpy as np
import pandas as pd

def recode_variables(df):
    """Convert certain fields with strings into fields with integers.

    Returns
    -------
    df : dataframe with re-coded variables added to it.
    """
    # Map gender column Female = > 0 and Male => 1
    df['Gender'] = df['Sex'].map( {'female':0 ,'male':1} ).astype(int)
    #df['Embarked_class'] = df['Embarked'].map( {'C':0, 'Q':1, 'S':2})
    
    embarked_dummy_vars = pd.get_dummies(df['Embarked'], prefix="Embarked_")
    df = pd.concat([df, embarked_dummy_vars], axis=1)
    #df = df.drop(['Embarked', 'Embarked_S'], axis=1)
    df = df.drop(['Embarked'], axis=1)
    
    return df


def replace_missing_age_values(df):
    """Replace missing age values with the median age for people with the same gender and class.
    
    """
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


def add_title_variable(df):
    # Get Title from Name column
    df['Title'] = df['Name'].str.split(',').str.get(1)
    
    #clean up Title column
    df['Title'] = df['Title'].map(str.strip)
    df['Title'] = df['Title'].str.split(' ').str.get(0)
    df['Title'] = df['Title'].map(str.strip)
    
    #Map titles to numbers
    df['Title_Code'] = pd.Categorical.from_array(df.Title).labels
    #df.get_dummies()
    return df

    

def replace_missing_age_values_using_titles(df):
    num_types_titles = int(len(df['Title'].value_counts()))

    medAges = {}
    for g in range(0,2):
        for c in range(1,4):
            for t in range(0,num_types_titles +1):
                key = str(g) + str(c) + str(t)
                medAges[key] = df[  (df['AgeIsNull']==0) & (df['Gender']==g) & (df['Pclass']==c)  & (df['Title_Code']==t) ]['Age'].median()
    
    import math
    
    df['newAgeFill'] = df['AgeFill']


    for g in range(0,2):
        for c in range(1,4):
            for t in range(0,num_types_titles +1):
                key = str(g) + str(c) + str(t)
                if math.isnan(medAges.get(key))==False:
                    print " Adding %d" %medAges.get(key)
                    df.loc[  (df['AgeIsNull']==1) & (df['Gender']==g) & (df['Pclass']==c)  & (df['Title_Code']==t) ,'newAgeFill'] = medAges.get(key)

    df.drop(['AgeFill'],axis=1)
    
    return df



if __name__ == '__main__':
    pass