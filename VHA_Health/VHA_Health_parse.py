import numpy as np
import pickle
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import json

def csv_data(prefix, info):
    out=[]
    for i in range(len(info)):
        print(i)
        out.append(pd.read_csv(prefix+info[i]+'.csv'))
    return out

def find_patient_col(csv):
    cols=csv.columns
    for i in range(len(cols)):
        if cols[i]=='PATIENT':
            return i
    for i in range(len(cols)):
        if cols[i]=='Id':
            return i
    return -1

def organize(info, unorg):
    out={i:dict() for i in unorg[3].Id}
    for i in range(unorg.shape[0]):
        print(unorg[i])
        id_col=find_patient_col(unorg[i])
        if id_col<0:
            continue
        cols=unorg[i].columns
        arr=np.array(unorg[i])
        for j in arr:
            patient=j[id_col]
            if patient not in out:
                break
            for k in range(j.size):
                if cols[k]=='PATIENT' or cols[k]=='Id':
                    continue
                key=info[i]+'_'+cols[k]
                if key not in out[patient]:
                    out[patient][key]=[]
                out[j[id_col]][key].append(j[k])
        print(i)
    return out

def convert_to_date(str_date):
    arr=[int(i) for i in str_date.split('-')]
    return datetime.date(arr[0], arr[1], arr[2])

def convert_to_time(start_str, end_str):
    start_str=start_str.replace('T', '-')
    start_str=start_str.replace(':', '-')
    start_str=start_str.replace('Z', '')
    end_str=end_str.replace('T', '-')
    end_str=end_str.replace(':', '-')
    end_str=end_str.replace('Z', '')
    return ((datetime.datetime(*[int(i) for i in end_str.split('-')])-
            datetime.datetime(*[int(i) for i in start_str.split('-')])).total_seconds()/86400)

def convert_ages(deathdate):
    if deathdate!=deathdate:
        return datetime.date.today()
    return convert_to_date(deathdate)

'''            
#mac            
train_prefix='/Users/johnathanxie/Documents/Python/datasets/VHA_Health/train/'
test_prefix='/Users/johnathanxie/Documents/Python/datasets/VHA_Health/test/'
'''


train_prefix=r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train'
test_prefix=r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\test'
prefix=train_prefix

allergies=pd.read_csv(os.path.join(prefix, 'allergies.csv'))
care_plans=pd.read_csv(os.path.join(prefix, 'careplans.csv'))
conditions=pd.read_csv(os.path.join(prefix, 'conditions.csv'))
patients=pd.read_csv(os.path.join(prefix, 'patients.csv'))
imaging_studies=pd.read_csv(os.path.join(prefix, 'imaging_studies.csv'))
immunizations=pd.read_csv(os.path.join(prefix, 'immunizations.csv'))
observations=pd.read_csv(os.path.join(prefix, 'observations.csv'))
organizations=pd.read_csv(os.path.join(prefix, 'organizations.csv'))
encounters=pd.read_csv(os.path.join(prefix, 'encounters.csv'))
devices=pd.read_csv(os.path.join(prefix, 'devices.csv'))
supplies=pd.read_csv(os.path.join(prefix, 'supplies.csv'))
procedures=pd.read_csv(os.path.join(prefix, 'procedures.csv'))
medications=pd.read_csv(os.path.join(prefix, 'medications.csv'))
payers=pd.read_csv(os.path.join(prefix, 'payers.csv'))
providers=pd.read_csv(os.path.join(prefix, 'providers.csv'))

deathdates=list(patients.DEATHDATE)
#for training
valid_deathdates=set(i for i in deathdates if (i==i and convert_to_date(i).year>2019))
patient_IDs=list(patients.Id)
patient_IDs=[patient_IDs[i] for i in range(len(patient_IDs)) if (deathdates[i]!=deathdates[i] or
                                                    deathdates[i] in valid_deathdates
                                                    and patient_IDs[i]==patient_IDs[i])]
#for training add or if deathdates in valid deathdates

all_data=np.array([allergies, imaging_studies, conditions, patients, observations, care_plans, encounters,
                   devices, supplies, procedures, medications, immunizations, supplies, payers, providers])
info=['allergies', 'imaging_studies', 'conditions', 'patients', 'observations', 'care_plans', 'encounters',
      'devices', 'supplies', 'procedures', 'medications', 'immunizations', 'supplies', 'payers', 'providers']

#creating y_train_data
covid_patient_ids=conditions[conditions.CODE==840539006].PATIENT.unique()
negative_covid_patient_ids=observations[(observations.CODE == '94531-1') &
                                        (observations.VALUE == 'Not detected (qualifier value)')].PATIENT.unique()
#negative_covid_patient_ids=[i for i in negative_covid_patient_ids if i not in covid_patient_ids]
inpatient_ids=encounters[(encounters.REASONCODE==840539006) & (encounters.CODE == 1505002)].PATIENT
deceased_ids=np.intersect1d(covid_patient_ids, patients[patients.DEATHDATE.notna()].Id)
vent_ids=procedures[(procedures.CODE==26763009) & (procedures.PATIENT.isin(covid_patient_ids))].PATIENT
icu_ids=encounters[(encounters.CODE==305351004) & (encounters.PATIENT.isin(covid_patient_ids))].PATIENT

'''
COVID_19=dict()
for i in covid_patient_ids:
    COVID_19[i]=1

for i in negative_covid_patient_ids:
    COVID_19[i]=0
'''

COVID_19={i:0 for i in patient_IDs}
for i in covid_patient_ids:
    COVID_19[i]=1


#(encounters.REASONCODE==840539006) & (encounters.CODE == 1505002)

hospitalized_days={i:0 for i in patient_IDs}
for i, j, k in zip(encounters[(encounters.REASONCODE==840539006) & (encounters.CODE == 1505002)
                              & (encounters.PATIENT.isin(patient_IDs))].PATIENT,
                   encounters[(encounters.REASONCODE==840539006) & (encounters.CODE == 1505002)
                              & (encounters.PATIENT.isin(patient_IDs))].START,
                   encounters[(encounters.REASONCODE==840539006) & (encounters.CODE == 1505002)
                              & (encounters.PATIENT.isin(patient_IDs))].STOP):
    hospitalized_days[i]+=convert_to_time(j, k)

deceased_patients={i:1 for i in patient_IDs}
for i in deceased_ids:
    deceased_patients[i]=0

vent_status={i:0 for i in patient_IDs}
for i in vent_ids:
    vent_status[i]=1

icu_days={i:0 for i in patient_IDs}
for i, j, k in zip(encounters[(encounters.CODE==305351004) & (encounters.PATIENT.isin(covid_patient_ids))
                              & (encounters.PATIENT.isin(patient_IDs))].PATIENT,
                   encounters[(encounters.CODE==305351004) & (encounters.PATIENT.isin(covid_patient_ids))
                              & (encounters.PATIENT.isin(patient_IDs))].START,
                   encounters[(encounters.CODE==305351004) & (encounters.PATIENT.isin(covid_patient_ids))
                              & (encounters.PATIENT.isin(patient_IDs))].STOP):
    icu_days[i]+=convert_to_time(j, k)

y_train=[COVID_19, hospitalized_days, deceased_patients, vent_status, icu_days]
y_train_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\y_dict', 'w')
json.dump(y_train, y_train_file)
y_train_file.close()

#deleting target data from patient info
conditions=conditions.drop(conditions[conditions.PATIENT.isin(covid_patient_ids)].index)
observations=observations.drop(observations[observations.PATIENT.isin(negative_covid_patient_ids)].index)
encounters=encounters.drop(encounters[encounters.PATIENT.isin(inpatient_ids)].index)
#patients=patients.drop(patients[patients.Id.isin(deceased_ids)].index)
procedures=procedures.drop(procedures[procedures.PATIENT.isin(vent_ids)].index)
encounters=encounters.drop(encounters[encounters.PATIENT.isin(icu_ids)].index)

patient_data=organize(info, all_data)
patient_data_in_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\patient_data', 'r')
patient_data=json.load(patient_data_in_file)
patient_data_in_file.close()
patient_data_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\patient_data', 'w')
patient_IDs_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\patient_IDs', 'w')
json.dump(patient_data, patient_data_file)
json.dump(patient_IDs, patient_IDs_file)
patient_data_file.close()
patient_IDs.close()





types_of_each=dict()
for i in range(len(all_data)):
    print(i)
    for j, k in zip(all_data[i], all_data[i].columns):
        types_of_each[info[i]+k]=list(set(j))
'''
keys=set()
for i in patient_IDs:
    patient_keys=patient_data[i].keys()
    for j in patient_keys:
        keys.add(j)

keys=list(keys)
for i in keys:
    if 'Id' in i:
        continue
    types_of_each[i]=set()

for i in range(len(patient_IDs)):
    patient=patient_data[patient_IDs[i]]
    stats=[a for a in patient.keys() if 'Id' not in a]
    for j in stats:
        patient[j]=list(a for a in patient[j] if a==a)
        if 'DATE' in j:
            patient[j]=list(convert_to_date(a) for a in patient[j])

for i in range(len(patient_IDs)):
    patient=patient_data[patient_IDs[i]]
    stats=[a for a in patient.keys() if 'Id' not in a]
    for j in stats:
        temp=patient[j]
        for k in temp:
            types_of_each[j].add(k)

for i in keys:
    types_of_each[i]=list(types_of_each[i])
'''

'''1 is true, 0 is false'''
#add HIV status and blood data found in observations.csv
x_train_columns=['age',  'gender', 'Asian', 'White', 'Black', 'HIV', 'stroke', 
                 'cardiac_arrest', 'healthcare coverage', 'healthcare expenses',
                 'has chronic lung disease(0 for none, 1 for small, 2 for non-small)', 
                 'has severe asthma', 'has serious heart conditions',  'BMI(>30 is 1, >40 is 2',
                 'has diabetes', 'has chronic kidney disease', 'has liver disease']
#finding patient_ids with certain info
ages={i:(convert_ages(k)-convert_to_date(j)).days/365.25 for i, j, k in
      zip(patients[patients.Id.isin(patient_IDs)].Id,
          patients[patients.Id.isin(patient_IDs)].BIRTHDATE,
          patients[patients.Id.isin(patient_IDs)].DEATHDATE)}

gender={i:0 for i in patient_IDs}
female_IDs=patients[(patients.GENDER=='F') & (patients.Id.isin(patient_IDs))].Id
for i in female_IDs:
    gender[i]=1

asian={i:0 for i in patient_IDs}
asian_IDs=patients[(patients.RACE=='asian') & (patients.Id.isin(patient_IDs))].Id
for i in asian_IDs:
    asian[i]=1

white={i:0 for i in patient_IDs}
white_IDs=patients[(patients.RACE=='white') & (patients.Id.isin(patient_IDs))].Id
for i in white_IDs:
    white[i]=1

black={i:0 for i in patient_IDs}
black_IDs=patients[(patients.RACE=='black') & (patients.Id.isin(patient_IDs))].Id
for i in black_IDs:
    black[i]=1

healthcare_coverage={i:j for i, j in zip(
    patients[patients.Id.isin(patient_IDs)].Id,
    patients[patients.Id.isin(patient_IDs)].HEALTHCARE_COVERAGE)}
healthcare_expenses={i:j for i, j in zip(
    patients[patients.Id.isin(patient_IDs)].Id,
    patients[patients.Id.isin(patient_IDs)].HEALTHCARE_EXPENSES)}

HIV_patients=observations[(observations.DESCRIPTION=='HIV status') &
                          (observations.VALUE=='HIV positive') &
                          (observations.PATiENT.isin(patient_IDs))].PATIENT.unique()
HIV={i:0 for i in patient_IDs}
for i in HIV_patients:
    HIV[i]=1

stroke_patients=conditions[(conditions.DESCRIPTION=='stroke') &
                           (conditions.PATIENT.isin(patient_IDs))].PATIENT
stroke={i:0 for i in patient_IDs}
for i in stroke_patients:
    stroke[i]=1

cardiac_arrest_signs=['History of cardiac arrest (situation)', 'Cardiac Arrest']
cardiac_arrest_patients=conditions[conditions.DESCRIPTION.isin(cardiac_arrest_signs)
                                   & (conditions.PATIENT.isin(patient_IDs))].PATIENT
cardiac_arrest={i:0 for i in patient_IDs}
for i in cardiac_arrest_patients:
    cardiac_arrest[i]=1

lung_disease=dict()
for i in patient_IDs:
    lung_disease[i]=0

small_lung_disease_descriptions=['Small cell carcinoma of lung (disorder)',
                                 'Primary small cell malignant neoplasm of lung  TNM stage 1 (disorder)',
                                 'Suspected lung cancer (situation)']
non_small_lung_disease_descriptions=['Non-small cell carcinoma of lung  TNM stage 1 (disorder)',
                                    'Non-small cell lung cancer (disorder)',
                                    'Non-small cell carcinoma of lung  TNM stage 2 (disorder)']
small_lung_disease_ids=conditions[(conditions.DESCRIPTION.isin(small_lung_disease_descriptions))
                                  & (conditions.STOP!=conditions.STOP)
                                   & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
non_small_lung_disease_ids=conditions[(conditions.DESCRIPTION.isin(non_small_lung_disease_descriptions))
                                      & (conditions.STOP!=conditions.STOP)
                                       & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
for i in small_lung_disease_ids:
    lung_disease[i]=1

for i in non_small_lung_disease_ids:
    lung_disease[i]=2

asthma_patient_IDs=conditions[(conditions.DESCRIPTION=='Childhood asthma')
                              & (conditions.STOP!=conditions.STOP)
                              & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
asthma={i:0 for i in patient_IDs}
for i in asthma_patient_IDs:
    asthma[i]=1

heart_condition_descriptions=['Injury of heart (disorder)', 'Chronic congestive heart failure (disorder)']
heart_conditions_IDs=conditions[(conditions.DESCRIPTION.isin(heart_condition_descriptions))
                                & (conditions.STOP!=conditions.STOP)
                                & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
heart_conditions={i:0 for i in patient_IDs}
for i in heart_conditions_IDs:
    heart_conditions[i]=1

obese_ids=conditions[(conditions.DESCRIPTION=='Body mass index 30+ - obesity (finding)')
                     & (conditions.STOP!=conditions.STOP)
                     & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
severly_obese_ids=conditions[(conditions.DESCRIPTION=='Body mass index 40+ - severely obese (finding)')
                             & (conditions.STOP!=conditions.STOP)
                             & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
obesity={i:0 for i in patient_IDs}
for i in obese_ids:
    obesity[i]=1

for i in severly_obese_ids:
    obesity[i]=2

diabetes_descriptions=['Macular edema and retinopathy due to type 2 diabetes mellitus (disorder)',
                       'Blindness due to type 2 diabetes mellitus (disorder)',
                       'Proliferative diabetic retinopathy due to type II diabetes mellitus (disorder)',
                       'Nonproliferative diabetic retinopathy due to type 2 diabetes mellitus (disorder)',
                       'Diabetic retinopathy associated with type II diabetes mellitus (disorder)', 'Prediabetes',
                       'Microalbuminuria due to type 2 diabetes mellitus (disorder)',
                       'Neuropathy due to type 2 diabetes mellitus (disorder)',
                       'Proteinuria due to type 2 diabetes mellitus (disorder)']
diabetes_ids=conditions[(conditions.DESCRIPTION.isin(diabetes_descriptions))
                        & (conditions.STOP!=conditions.STOP)
                        & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
diabetes={i:0 for i in patient_IDs}
for i in diabetes_ids:
    diabetes[i]=1

kidney_one_descriptions=['Chronic kidney disease stage 2 (disorder)', 'Injury of kidney (disorder)']
kidney_one_ids=conditions[(conditions.DESCRIPTION.isin(kidney_one_descriptions))
                          & (conditions.STOP!=conditions.STOP)
                          & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
kidney_two_ids=conditions[(conditions.DESCRIPTION=='Chronic kidney disease stage 2 (disorder)')
                          & (conditions.STOP!=conditions.STOP)
                          & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
kidney_three_ids=conditions[(conditions.DESCRIPTION=='Chronic kidney disease stage 3 (disorder)')
                            & (conditions.STOP!=conditions.STOP)
                            & (conditions.PATIENT.isin(patient_IDs))].PATIENT.unique()
kidney_disease={i:0 for i in patient_IDs}
for i in kidney_one_ids:
    kidney_disease[i]=1

for i in kidney_two_ids:
    kidney_disease[i]=2

for i in kidney_three_ids:
    kidney_disease[i]=3

x_train=[ages, gender, asian, white, black, HIV, stroke, cardiac_arrest, 
         healthcare_coverage, healthcare_expenses, lung_disease, asthma,
         heart_conditions, obesity, diabetes, kidney_disease]
x_test_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\test\x_dict', 'w')
x_test_file.close()
x_train_file=open(r'C:\Users\johna\OneDrive\Documents\Python\Datasets\VHA_health\train\x_dict', 'w')
json.dump(x_train, x_train_file)
x_train_file.close()

