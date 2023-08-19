import numpy as np
import pandas as pd
from datetime import datetime
import shap
import lime
from scipy.stats import linregress
import lime.lime_tabular
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from keras.layers import Dropout 
import matplotlib.pyplot as plt 
from tensorflow.keras.optimizers import Adam
from keras.layers import GRU 
from keras.layers import Flatten 
from keras.layers.convolutional import Conv1D, MaxPooling1D


def predictWithTrainedModel(data):
    pred = []
    for x in data:
        test_data = np.reshape(x, (1, x.shape[0], 1))
        aqi = selected_model.predict(test_data)
        pred.append(aqi[0][0])

    return np.array(pred)

def get_prediction_from_model(dataset):
    pred = selected_model.predict(dataset)

    return np.array(pred)

def get_lime_values(dataset):
    lime_values = []
    single_lime_value = []
    lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(training_set, feature_names=features, class_names=['AQI'], verbose=True, mode='regression')

    for test_data in dataset:
      test_data = np.reshape(test_data, (1, test_data.shape[0],test_data.shape[1]))
      lime_exp = lime_explainer.explain_instance(test_data, get_prediction_from_model, num_features=len(features), labels=(1,))

      coeffs = lime_exp.local_exp[0]
      single_lime_value = [0] * len(features)
      for c in coeffs:
          single_lime_value[c[0]] += c[1]

      lime_values.append(single_lime_value)

    return np.array(lime_values)

def XAIEvaluation(model, model_name):
    global selected_model
    # Create a shap explainer
    selected_model = model
    custom_training_set = np.reshape(training_set, (training_set.shape[0], training_set.shape[1], 1))
    
    # shap_explainer = shap.KernelExplainer(get_prediction_from_model, custom_training_set)
    shap_explainer = shap.KernelExplainer(predictWithTrainedModel, custom_training_set)
    
    X_test = dataset_x[num_trained_item:]
    y_test = dataset_y[num_trained_item:]
    
    
    
    num_items = 200
    subset_A = X_test[:num_items]
    subset_B = X_test[num_items:2*num_items]
    
    subset_A = np.reshape(subset_A, (subset_A.shape[0], subset_A.shape[1], 1))
    subset_B = np.reshape(subset_B, (subset_B.shape[0], subset_B.shape[1], 1))
    
    shap_values_A = shap_explainer.shap_values(subset_A)
    shap_values_B = shap_explainer.shap_values(subset_B)
    
    shap_values = np.array([*shap_values_A, *shap_values_B])
    
    
    """### LIME Model"""
    
    
    num_items = 200
    subset_A = X_test[:num_items]
    subset_B = X_test[num_items:2*num_items]
    
    lime_values_A = get_lime_values(subset_A)
    lime_values_B = get_lime_values(subset_B)
    
    lime_values = np.array([*lime_values_A, *lime_values_B])
    
    # Calculate feature coverage
    shap_coverages = []
    
    nonzero_counts = np.sum(shap_values != 0, axis=0)
    nonzero_percents = nonzero_counts / shap_values.shape[0]
    shap_coverages.extend(nonzero_percents)
    
    
    
    # Calculate feature coverage
    lime_coverages = []
    
    nonzero_counts = np.sum(lime_values != 0, axis=0)
    nonzero_percents = nonzero_counts / lime_values.shape[0]
    lime_coverages.extend(nonzero_percents)
    
    
    
    # calculate the average absolute SHAP value for each feature
    avg_shap = np.abs(shap_values).mean(axis=0)
    
    
    """### LIME Model"""
    
    # calculate the average absolute lime value for each feature
    avg_lime = np.abs(lime_values).mean(axis=0)
    
    
    shap_consistency = np.mean(np.abs(shap_values_A.flatten() - shap_values_B.flatten()))
    
    
    slope, intercept, r_value, p_value, std_err = linregress(shap_values_A.flatten(), shap_values_B.flatten())
    shap_icc = (r_value**2) * (np.sum(shap_values_A.flatten()**2) / len(shap_values_A.flatten()))
    
    """### LIME Model"""
    
    lime_consistency = np.mean(np.abs(lime_values_A.flatten() - lime_values_B.flatten()))    
    
    slope, intercept, r_value, p_value, std_err = linregress(lime_values_A.flatten(), lime_values_B.flatten())
    lime_icc = (r_value**2) * (np.sum(lime_values_A.flatten()**2) / len(lime_values_B.flatten()))
    
    
    
    # Errors
    
    y_predict = model.predict(X_test)
    
    mae = mean_absolute_error(y_predict, y_test)
    mse = mean_squared_error(y_predict, y_test)
    rmse = math.sqrt(mse)
    
    
    row = {"Model Name": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, 
           "SHAP Coverage":shap_coverages, "SHAP Avg Coverage": np.mean(shap_coverages), "SHAP Relevence":avg_shap, "SHAP Avg Relevence":np.mean(avg_shap), "SHAP CE":shap_consistency, "SHAP ICC":shap_icc,
            "LIME Coverage":lime_coverages, "LIME Avg Coverage": np.mean(lime_coverages), "LIME Relevence":avg_lime, "LIME Avg Relevence":np.mean(avg_lime), "LIME CE":lime_consistency, "LIME ICC":lime_icc}
    output_file.append(row)
    pd.DataFrame.from_dict(output_file).to_csv(output_file_path, index=False)


root_path = '..'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values
training_set = dataset_x[:500]
num_trained_item = 10000

x_train,x_test,y_train,y_test=train_test_split(dataset_x[:num_trained_item], dataset_y[:num_trained_item], test_size=0.2, random_state=42)


file_name = "ml_"+str(datetime.now())+".csv"
file_name = file_name.replace(' ', '-')
file_name = file_name.replace(':', '-')
output_file_path = root_path+"/result/"+file_name
output_file = []

"""# SHAP Explanation"""

start = datetime.now()

num_epochs = 5

#Scale the all of the data to be values between 0 and 1 
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
scaled_xtrain = X_scaler.fit_transform(x_train) 
scaled_xtest = X_scaler.fit_transform(x_test) 

#Convert to numpy arrays 
scaled_xtrain, scaled_ytrain = np.array(scaled_xtrain), np.array(y_train)

#Reshape the data into 3-D array 
scaled_x = np.reshape(scaled_xtrain, (scaled_xtrain.shape[0],scaled_xtrain.shape[1],1)) 
scaled_y = np.reshape(scaled_ytrain, (scaled_ytrain.shape[0],1))

#Convert x_test to a numpy array 
scaled_xtest = np.array(scaled_xtest)

#Reshape the data into 3-D array 
scaled_xtest = np.reshape (scaled_xtest, (scaled_xtest.shape[0],scaled_xtest.shape[1],1))


# LSTM
model_lstm = Sequential()
# model_lstm.add(LSTM(units = 108, return_sequences = False, input_shape = (scaled_x.shape[1], 1))) 
model_lstm.add(LSTM(units = 108, return_sequences = False, input_shape = (x_train.shape[1],))) 
model_lstm.add(Dropout(0.2)) 
model_lstm.add(Dense(units = 1))

#compile and fit the model on 40 epochs 
optimizer = Adam(lr=0.0001) 
model_lstm.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=['accuracy']) 
# lstm_history = model_lstm.fit(scaled_x, scaled_y, epochs = num_epochs, batch_size = 32, shuffle=True, validation_split=0.20)
lstm_history = model_lstm.fit(x_train, y_train, epochs = num_epochs, batch_size = 32, shuffle=True, validation_split=0.20)

#check predicted values 
predictions = model_lstm.predict(scaled_xtest) 
x_pred = model_lstm.predict(scaled_x) 


# List all data in history 
print(lstm_history.history.keys() )
print("\n") 
# summarize history for accuracy 
plt.plot(lstm_history.history['loss']) 
plt.plot(lstm_history.history['val_loss']) 
plt.title('LSTM Learning Curve') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt. legend(['train', 'test'], loc='upper left') 
plt.show()


selected_model = model_lstm
custom_training_set = np.reshape(training_set, (training_set.shape[0], training_set.shape[1], 1))

# shap_explainer = shap.KernelExplainer(predictWithTrainedModel, custom_training_set)
shap_explainer = shap.DeepExplainer(model_lstm, custom_training_set)

X_test = dataset_x[num_trained_item:]
y_test = dataset_y[num_trained_item:]


num_items = 200
subset_A = X_test[:num_items]
subset_B = X_test[num_items:2*num_items]

subset_A = np.reshape(subset_A, (subset_A.shape[0], subset_A.shape[1], 1))
subset_B = np.reshape(subset_B, (subset_B.shape[0], subset_B.shape[1], 1))

test_data = subset_A[0]
shap_values_A = shap_explainer.shap_values(test_data)

shap_values_A = shap_explainer.shap_values(subset_A)
shap_values_B = shap_explainer.shap_values(subset_B)

shap_values = np.array([*shap_values_A, *shap_values_B])

XAIEvaluation(model_lstm, "LSTM")

# GRU
model_gru = Sequential() 
model_gru.add(GRU (100, return_sequences = False, input_shape = (scaled_x.shape[1], 1))) 
model_gru.add(Dropout(0.2)) 
model_gru.add(Dense(units=1)) 
optimizer = Adam(lr=0.0001) 
model_gru.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=['accuracy']) 
gru_history = model_gru.fit(scaled_x, scaled_y, epochs = num_epochs, batch_size = 32, shuffle=True, validation_split=0.20)


#check predicted values 
predictions = model_gru.predict(scaled_xtest) 
x_pred = model_gru.predict(scaled_x)


# List all data in history 
print(gru_history.history.keys() )
print("\n") 
# summarize history for accuracy 
plt.plot(gru_history.history['loss']) 
plt.plot(gru_history.history['val_loss']) 
plt.title('GRU Learning Curve') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt. legend(['train', 'test'], loc='upper left') 
plt.show()



# CNN
model_cnn = Sequential() 
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(scaled_x.shape [1], 1))) 
model_cnn.add(MaxPooling1D(pool_size=2)) 
model_cnn.add(Flatten())
model_cnn.add(Dense(180, activation='relu')) 
model_cnn.add(Dense(1))
model_cnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) 

# fit model 
cnn_history = model_cnn.fit(scaled_x, scaled_y, epochs=num_epochs, verbose=1, validation_split=0.20) 


#check predicted values 
predictions = model_cnn.predict(scaled_xtest) 
x_pred = model_cnn.predict(scaled_x) 


# List all data in history 
print(cnn_history.history.keys() )
print("\n") 
# summarize history for accuracy 
plt.plot(cnn_history.history['loss']) 
plt.plot(cnn_history.history['val_loss']) 
plt.title('CNN Learning Curve') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt. legend(['train', 'test'], loc='upper left') 
plt.show()


XAIEvaluation(model_lstm, "LSTM")
XAIEvaluation(model_gru, "GRU")
XAIEvaluation(model_cnn, "CNN")