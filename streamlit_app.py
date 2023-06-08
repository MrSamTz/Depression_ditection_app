# import important packages

import streamlit as st
import joblib
import pandas as pd
from os.path import dirname, join, realpath
import sklearn
# from streamlit.cache import st.cache_resource, st.cache_data

# add banner image
st.title("Depression Forecast: A Technology-Driven Approach to Mental Health Prediction")
st.image("Image/download.jpg", width=530)
st.subheader(
    """
A simple app that predicts if an individual suffers depression.
"""
)

# form to collect user information 

our_form = st.form(key="depression_form")
 
age = our_form.number_input("Enter your age", min_value=1, max_value=100) 
married = our_form.selectbox("Are you married?",("Yes","No"))
edu = our_form.number_input("How many years of education?", min_value=1,max_value=30)
durable_investment = our_form.number_input("How much is your durable investment?", min_value = 0, max_value = 10000000)
nondurable_investment = our_form.number_input("How much is your non durable investment?", min_value = 0, max_value = 10000000)
ent_employees = our_form.number_input("How many Employees do you have?", min_value=0,max_value=100)
ent_total_cost = our_form.number_input("How much do you spend on every month?", min_value=0,max_value=1000)
amount_saved_mpesa = our_form.number_input("How much do you save in M-PESA?", min_value= 0, max_value= 10000000)
cons_alcohol = our_form.number_input("How much do you spend on alcohol?", min_value=0,max_value=1000)
fs_adwholed_often = our_form.number_input("How many days without eating?", min_value=0,max_value=30), 

submit = our_form.form_submit_button(label="make prediction")


# load the model and scaler

with open(
    join(dirname(realpath(__file__)), "Model/ModelHist.pkl"),
    "rb",
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "Preprocessing/Preprocessing.pkl"), "rb"
) as f:
    scaler = joblib.load(f)
    
def femaleres_transform(value):
	if value=='Female':
		return 1
	else: 
		return 0
		
def married_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
		
def saved_mpesa_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0

def preprocessing_data(data, scaler):
    # scale our data into range of 0 and 1
    data = scaler.transform(data.values.reshape(-1, 1))
    feat_col = ['age', 'ent_total_cost', 'durable_investment',
                'nondurable_investment', 'ent_employees', 'married',
                'edu', 'cons_alcohol', 'fs_adwholed_often', 'amount_saved_mpesa']

    return pd.DataFrame(data.reshape(-1, 10), columns=feat_col)


@st.cache_data()
def cached_preprocessing_data(data, scaler):
    return preprocessing_data(data, scaler)

if submit:


    # collect inputs
    input = {
        "age": age,
        "married": married_transform(married), 
        "ent_employees": ent_employees, 
        "cons_alcohol":cons_alcohol,
        "fs_adwholed_often":fs_adwholed_often,
        "durable_investment":durable_investment,
        "nondurable_investment":nondurable_investment,
         "ent_total_cost":ent_total_cost,
         "amount_saved_mpesa":amount_saved_mpesa,
        "edu":edu
       
    }

    # create a dataframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data, scaler=scaler)

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the RFC task
    st.header("Results")
    if output == 1:
        st.write(
            "You are likely to be depressed with probability of {} 😔 Please visit nearest doctor for counselling.".format(
                probability
            )
        )
    elif output == 0:
        st.write(
            "You are likely not to be depressed with probability of {} 😊".format(
                probability
            )
        )
        
        
url = "https://twitter.com/@AbediModest"
st.write("Developed with @ by [Finalist 2022/2023](%s)" % url)
