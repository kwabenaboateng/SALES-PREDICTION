import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn 
from PIL import Image



# Load the saved components:
with open("dt_model.pkl", "rb") as f:
    components = pickle.load(f)

# Extract the individual components
num_imputer = components["num_imputer"]
cat_imputer = components["cat_imputer"]
encoder = components["encoder"]
scaler = components["scaler"]
dt_model = components["models"]

# Create the app

st.set_page_config(
    layout="wide"
)


# Add an image or logo to the app
image = Image.open('copofav.jpg')

# Open the image file
st.image(image)


#add app title
st.title("SALES PREDICTION APP")


# Add some text
st.write("Please ENTER the relevant data and CLICK Predict.")

 # Create the input fields
input_data = {}
col1,col2,col3 = st.columns(3)
with col1:
    input_data['store_nbr'] = st.slider("Store Number",0,54)
    input_data['products'] = st.selectbox("Products Family", ['OTHERS', 'CLEANING', 'FOODS', 'STATIONERY', 'GROCERY', 'HARDWARE',
       'HOME', 'CLOTHING'])
    input_data['onpromotion'] =st.number_input("Discount Amt On Promotion",step=1)
    input_data['state'] = st.selectbox("State", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
       'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
       'El Oro', 'Esmeraldas', 'Manabi'])
with col2:    
    input_data['store_type'] = st.selectbox("Store Type",['D', 'C', 'B', 'E', 'A'])
    input_data['cluster'] = st.number_input("Cluster",step=1)
    input_data['dcoilwtico'] = st.number_input("DCOILWTICO",step=1)
    input_data['year'] = st.number_input("Year to Predict",step=1)
with col3:    
    input_data['month'] = st.slider("Month",1,12)
    input_data['day'] = st.slider("Day",1,31)
    input_data['dayofweek'] = st.number_input("Day of Week,0=Sunday and 6=Satruday",step=1)
    input_data['end_month'] = st.selectbox("Is it End of the Month?",['True','False'])


  # Create a button to make a prediction
if st.button("Predict"):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # categorizing the products
    food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
    home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
    clothing_families = ['LINGERIE', 'LADYSWARE']
    grocery_families = ['GROCERY I', 'GROCERY II']
    stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
    cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
    hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']
    others_families = ['AUTOMOTIVE', 'BEAUTY','CELEBRATION', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',  'PET SUPPLIES']


 

    # Apply the same preprocessing steps as done during training
    input_df['products'] = np.where(input_df['products'].isin(food_families), 'FOODS', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(home_families), 'HOME', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(clothing_families), 'CLOTHING', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(grocery_families), 'GROCERY', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(stationery_families), 'STATIONERY', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(cleaning_families), 'CLEANING', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(hardware_families), 'HARDWARE', input_df['products'])
    input_df['products'] = np.where(input_df['products'].isin(others_families), 'OTHERS', input_df['products'])


    categorical_columns = ['products', 'end_month', 'store_type', 'state']
    numerical_columns =['store_nbr','onpromotion','cluster','dcoilwtico','year','month','day','dayofweek']
    # Impute missing values
    input_df_cat = input_df[categorical_columns].copy()
    input_df_num = input_df[numerical_columns].copy()
    input_df_cat_imputed = cat_imputer.transform(input_df_cat)
    input_df_num_imputed = num_imputer.transform(input_df_num)

    # Encode categorical features
    input_df_cat_encoded = pd.DataFrame(encoder.transform(input_df_cat_imputed).toarray(),
                                        columns=encoder.get_feature_names_out(categorical_columns))

    # Scale numerical features
    input_df_num_scaled = scaler.transform(input_df_num_imputed)
    input_df_num_sc = pd.DataFrame(input_df_num_scaled, columns=numerical_columns)

    # Combine encoded categorical features and scaled numerical features
    input_df_processed = pd.concat([input_df_num_sc, input_df_cat_encoded], axis=1)

    # Make predictions using the trained model
    predictions = dt_model.predict(input_df_processed)

    # Display the predicted sales value to the user:
    st.write("Predicted Sales:", predictions[0])
