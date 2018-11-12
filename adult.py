import pandas as pd
import numpy as np
data = pd.read_csv('adultdata.csv', header=None, sep=', ')
data.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
M = {'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'}
