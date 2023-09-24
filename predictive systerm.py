import numpy as np
import pickle

# loading the saved model

loaded_model=pickle.load(open(r'C:\Users\ailla\Downloads\Diabeties/trained_model.sav','rb'))

input_data=(5,166,72,19,175,25.8,0.587,51)

# changeing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshaping the array as we are predicting for one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)



prediction=loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
  print('the person is Not Diabetic')
else:
    print('the person is Diabetic')