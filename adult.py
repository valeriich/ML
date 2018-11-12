import pandas as pd
import numpy as np
data = pd.read_csv('adultdata.csv', header=None, sep=', ')
data.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
M = {'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'}
data['Married'] = 0
for i in data['MaritalStatus'].index:
    if data['MaritalStatus'][i] in M:
        data['Married'][i] = 'Mrd'
    else:
        data['Married'][i] = 'Nope'
