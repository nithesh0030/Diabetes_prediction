import numpy as np
import pickle
import streamlit as st
import prediction
import sklearn

# loading the saved model

loaded_model=pickle.load(open('trained_model.sav','rb'))

# creating a functio for prediction

def diabetes_prediction(input_data):

    # changeing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    # reshaping the array as we are predicting for one instance
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)



    prediction=loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
        return'the person is Not Diabetic'
    else:
        return'the person is Diabetic'
    



def main():


    # giving a title
    st.title('Diabetes Prediction web APP')

    # getting the input data from the user 

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')



    # code for prediction
    Diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        st.success(diagnosis)



if __name__=='__main__':
    main()
