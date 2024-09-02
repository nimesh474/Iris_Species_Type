from django.shortcuts import render
import numpy as np
import pickle

lgr_path = r'D:/DeerWalk/iris_flower_type_predication/ml_model/logistic_regression_model.pkl'
# load logistic regression model
with open(lgr_path, 'rb') as f:
    data = pickle.load(f)
    model_lgr = data['model_lgr']
    scaler = data['scaler']

svc_path = r'D:/DeerWalk/iris_flower_type_predication/ml_model/svc_model.pkl'
# load support vector machine classifier model
with open(svc_path, 'rb') as file:
    svc_data = pickle.load(file)
    model_svc = svc_data['model_svc']

# Iris Flower Type Prediction
def predict_type(request):
    if request.method == 'GET':
        return render(request, 'prediction_form.html')
    
    else:
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        values = [sepal_length, sepal_width, petal_length, petal_width]
        # reshape to 2D
        reshaped_values = np.array(values).reshape(1,-1)

        # standardize the values
        scaled_data = scaler.transform(reshaped_values)

        # print(scaled_data)

        # logistic regression prediction
        lgr_prediction = model_lgr.predict(scaled_data)
        
        # print(lgr_prediction)

        # support vector classifier prediction
        svc_prediction = model_svc.predict(scaled_data)

        # print(svc_prediction)

        context = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width,
            'lgr_prediction': lgr_prediction[0],
            'svc_prediction': svc_prediction[0]
        }

        return render(request, 'prediction_form.html', context)
