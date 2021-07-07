
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import math
import streamlit as st
st.title('Project- liver disease prediction using machine learning')


info = pd.read_csv('./dataset.csv')
info.head()
info.isna().sum() # Albumin_and_Globulin_Ratio    4 filling with mean()

info['Albumin_and_Globulin_Ratio'].isna().sum()
info['Albumin_and_Globulin_Ratio'].fillna(info['Albumin_and_Globulin_Ratio'].median(),inplace=True)
# info.isna().sum()

# info.info() # info

dt = tree.DecisionTreeClassifier()
# X = 
st.title("Indormation / Data sets information")
st.write(info.columns)


# In[38]:


#  rename 
info.rename(columns ={'Dataset':'Target'},inplace=True)
info.columns


# # In[81]:


X = info.drop('Target',axis=1)
y = info['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.title('Algorithm')
st.write(dt.fit(X_train,y_train))

# # dt.predict([[1,1,1,1,1,1,1,1,1,1]])
# # testing
# '''
# Age', 'sex', 'Total_Bilirubin', 'Direct_Bilirubin',
#        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#        'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
#        'Albumin_and_Globulin_Ratio'
# '''
st.title("Making predections...")

Sex = st.text_input("Entre your Sex (male = 1 and female = 2)")  # 1
age = st.text_input("Entre your age   ") # 2 
Total_Bilirubin = st.text_input("Entre your Total_Bilirubin >  ") # 3
Direct_Bilirubin = st.text_input("Entre your Direct_Bilirubin >  ")# 4
Alkaline_Phosphotase = st.text_input("Entre your Alkaline_Phosphotase >  ") # 5
Alamine_Aminotransferase = st.text_input("Entre your Alamine_Aminotransferase >  ") # 6
Aspartate_Aminotransferase = st.text_input("Entre your Aspartate_Aminotransferase >  ") # 7
Total_Protiens = st.text_input("Entre your Total_Protiens >  ")# 8
Albumin = st.text_input("Entre your Albumin >  ") # 9
Albumin_and_Globulin_Rati = st.text_input("Entre your Albumin_and_Globulin_Ratio >  ") # 10 

st.write(info.astype(int).info())

if st.button('Submit'):
    results = dt.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase
                      ,Total_Protiens,Albumin,Albumin_and_Globulin_Rati]])
    
    st.title('Check for Results')
    st.title('Predection')
    for final in results:
        if final == 1:
            st.write('\n you have a Liver Disease')
        else:
            st.write(' you do not have a Liver Disease')






# '''

# news = str(input())
# st.title('Make Predections')
# predection = st.text_area('Make Predections')
# # st.title('Make Predections')
# print(manual_testing(news))
# st.write(predection)
# '''