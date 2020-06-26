import numpy as np
import pickle
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import random
import json
import sklearn
random.seed(30)

def create_sample_weights(y_train, weights='balanced'):
    if weights=='balanced':
        count=np.zeros(shape=(len(set(y_train)), ))
        for i in y_train:
            count[int(i)]+=1
        total=sum(count)
        weights=[]
        for i in count:
            weights.append(total/i)
    else:
        weights=weights
    print(weights)
    sample_weights=[0 for i in y_train]
    for i in range(len(y_train)):
        sample_weights[i]=weights[int(y_train[i])]
    return sample_weights

def create_data(x_dict, y_dict):
    keys=[]
    x_keys=[i.keys() for i in x_dict]
    for i in range(len(y_dict)):
        temp_keys=list(set(y_dict[i].keys()).intersection(*x_keys))
        random.shuffle(temp_keys)
        keys.append(temp_keys)
    print([len(i) for i in keys])
    x_total=[]
    y_total=[]
    for i in range(len(y_dict)):
        temp_keys=keys[i]
        temp_x_train=np.zeros(shape=(len(temp_keys), len(x_dict)))
        temp_y_train=np.zeros(shape=(len(temp_keys), ))
        for j in range(len(temp_keys)):
            temp_y_train[j]=y_dict[i][temp_keys[j]]
            for k in range(len(x_dict)):
                temp_x_train[j][k]=x_dict[k][temp_keys[j]]
        x_total.append(temp_x_train)
        y_total.append(temp_y_train)
    return x_total, y_total

def prediction_aided_data(x_dict, y_dict, COVID_19_predictions):
    #if you are going to use true predictions, change COVID_19_predictions to y_data slice
    keys=list(y_dict[0].keys())
    COVID_positive_IDs=[keys[int(i)] for i in np.where(COVID_19_predictions==1)[0]]
    new_keys=[]
    x_total=[]
    y_total=[]
    for i in range(1, len(y_dict)):
        new_keys.append(list(set(y_dict[i].keys()).intersection(COVID_positive_IDs)))
    print([len(i) for i in new_keys])
    for i in range(1, len(y_dict)):
        temp_keys=new_keys[i-1]
        temp_x_train=np.zeros(shape=(len(temp_keys), len(x_dict)))
        temp_y_train=np.zeros(shape=(len(temp_keys), ))
        for j in range(len(temp_keys)):
            temp_y_train[j]=y_dict[i][temp_keys[j]]
            for k in range(len(x_dict)):
                temp_x_train[j][k]=x_dict[k][temp_keys[j]]
        x_total.append(temp_x_train)
        y_total.append(temp_y_train)
    return new_keys, x_total, y_total

def evaluate_classifiers(y_pred, y_true, positive_label):
    false_positives=0
    false_negatives=0
    negatives=0
    positives=0
    for i, j in zip(y_pred, y_true):
        if i!=positive_label:
            negatives+=1
            if i!=j:
                false_negatives+=1
        else:
            positives+=1
            if i!=j:
                false_positives+=1
    print('Precision: ' + str((positives-false_positives)/positives))
    print('Recall: ' + str((positives-false_positives)/
                           (positives-false_positives+false_negatives)))
    print('FPR: ' + str(false_positives/negatives))
    return [positives, negatives, false_positives, false_negatives]

def convert(data):
    for i in range(len(data)):
        if data[i]==1:
            data[i]=0
        elif data[i]==2:
            data[i]==1
    return data

def create_predictions(x_dict, COVID_19_model, days_hospitalized_model,
                       deceased_model, vent_model, icu_model):
    keys=list(set(x_dict[0].keys()).intersection(x_dict[1:].keys))
    keys_to_val=[keys[i]:i for i in range(len(keys))]
    x_test=np.zeros(len(keys), len(x_dict))
    for i in range(len(x_dict)):
        curr_dict=x_dict[i]
        for j in range(len(keys)):
            x_test[j][i]=curr_dict[keys]
    COVID_19_predictions=COVID_19_model.predict(x_test)
    COVID_positive_indicies=np.where(COVID_19_predictions==1)[0]
    x_test_positive=np.array([x_test[i] for i in COVID_positive_indicies])
    
x_dict_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\x_dict', 'r')
y_dict_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\y_dict', 'r')
x_dict=json.load(x_dict_file)
y_dict=json.load(y_dict_file)
x_dict_file.close()
y_dict_file.close()
x_data, y_data=create_data(x_dict, y_dict)

#usually 4 1 split when training
COVID_19_split=117959
COVID_19_model=GradientBoostingClassifier()
COVID_19_model.fit(x_data[0][:COVID_19_split], y_data[0][:COVID_19_split],
                   sample_weight=create_sample_weights(y_data[0][:COVID_19_split]))
print(COVID_19_model.score(x_data[0][COVID_19_split:], y_data[0][COVID_19_split:]))
COVID_19_predictions=COVID_19_model.predict(x_data[0][COVID_19_split:])
print(evaluate_classifiers(COVID_19_predictions, y_data[0][COVID_19_split:], 1))
COVID_19_probabilities=COVID_19_model.predict_proba(x_data[0][COVID_19_split:])[:, 1]
disp = sk.metrics.plot_precision_recall_curve(COVID_19_model, x_data[0][COVID_19_split:],  y_data[0][COVID_19_split:])
plt.show()

new_keys, pa_x_data, pa_y_data=prediction_aided_data(x_dict, y_dict, y_data[0])

days_hospitalized_split=73697
days_hospitalized_model=GradientBoostingRegressor(max_depth=3)
days_hospitalized_model.fit(pa_x_data[0][:days_hospitalized_split], pa_y_data[0][:days_hospitalized_split])
print(days_hospitalized_model.score(pa_x_data[0][days_hospitalized_split:], pa_y_data[0][days_hospitalized_split:]))
days_hospitalized_predictions=days_hospitalized_model.predict(pa_x_data[0][days_hospitalized_split:])

deceased_split=73697
deceased_model=GradientBoostingClassifier()
deceased_model.fit(pa_x_data[1][:deceased_split], pa_y_data[1][:deceased_split],
                   sample_weight=create_sample_weights(pa_y_data[1][:deceased_split]))
print(deceased_model.score(pa_x_data[1][deceased_split:], pa_y_data[1][deceased_split:]))
deceased_predictions=deceased_model.predict(pa_x_data[1][deceased_split:])
print(evaluate_classifiers(deceased_predictions, pa_y_data[1][deceased_split:], 1))
deceased_probabilities=COVID_19_probabilities=deceased_model.predict_proba(pa_x_data[1][deceased_split:])[:, 1]
disp = plot_precision_recall_curve(deceased_model, pa_x_data[1][deceased_split:],  pa_y_data[1][deceased_split:])
plt.show()

vent_split=73697
vent_model=GradientBoostingClassifier()
vent_model.fit(pa_x_data[2][:vent_split], pa_y_data[2][:vent_split])
print(vent_model.score(pa_x_data[2][vent_split:], pa_y_data[2][vent_split:]))
vent_predictions=vent_model.predict(pa_x_data[2][vent_split:])
print(evaluate_classifiers(vent_predictions, pa_y_data[2][vent_split:], 1))
vent_probabilities=vent_model.predict_proba(pa_x_data[2][vent_split:])[:, 1]
disp = plot_precision_recall_curve(deceased_model, pa_x_data[2][vent_split:],  pa_y_data[2][vent_split:])
plt.show()

icu_days_split=73697
icu_days_model=GradientBoostingRegressor()
icu_days_model.fit(pa_x_data[3][:icu_days_split], pa_y_data[3][:icu_days_split])
print(icu_days_model.score(pa_x_data[3][icu_days_split:], pa_y_data[3][icu_days_split:]))

