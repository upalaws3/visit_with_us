import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourism package being bought by a customer .
Enter the below feature values to get the prediction
""")

Age         =  st.number_input("Age", 0,100)
TypeofContact     = st.selectbox("Type of Contact", ["Self Enquiry","Company Invited"])
CityTier   = st.selectbox("City Tier", [1,2,3])
DurationOfPitch   = st.number_input("Duration of Pitch in mins" , 0,100)
Occupation    = st.selectbox("Occupation", ["Salaried","Free Lancer","Small Business","Large Business"])
Gender    = st.selectbox("Gender", ["Male","Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting",0,100)
NumberOfFollowups    = st.number_input("Number of Followups",0,100)
ProductPitched       = st.selectbox("Product Pitched", ["Deluxe","Basic","Standard","Super Deluxe","King"])
PreferredPropertyStar   = st.selectbox("Preferred Property Star",[3,4,5])
MaritalStatus    = st.selectbox("Marital Status", ["Single","Divorced","Married"])
NumberOfTrips   = st.number_input("Number of Trips",0,100)
Passport    = st.selectbox("Passport", [0,1])
PitchSatisfactionScore    = st.selectbox("Pitch Satisfaction Score" , [1,2,3,4,5])
OwnCar    = st.selectbox("Own a Car", [0,1])
NumberOfChildrenVisiting    = st.number_input("Number of Children visiting",0,25)
Designation     = st.selectbox("Designation",["Manager","Executive","Senior Manager","AVP","VP"])
MonthlyIncome = st.number_input("Monthly Income",0,100000)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("Predict Prod Taken"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Product Plan Purchased" if prediction == 1 else "Tourism Product Plan Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
