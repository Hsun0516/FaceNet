import os
import pandas as pd
import numpy as np
import random

data = [[file]+file.split('_')[:-1] for file in os.listdir('Data\\UTKFace')]
random.shuffle(data)
df = pd.DataFrame(data, columns=["file_name", "age", "gender", 'race'])
df = df.dropna(axis=0, how='any')
print(df.info())
print(df.head())
df.to_csv('data.csv')

file = [os.path.join(r".\Data\UTKFace", i) for i in df['file_name'].tolist()]
file_train = file[:14226]
file_validation = file[14226:18966]
file_test = file[18966:]
np.save('file_train.npy', file_train)
np.save('file_validation.npy', file_validation)
np.save('file_test.npy', file_test)

ages = genders = races = np.array([])
for file in df['file_name'].tolist():
    age, gender, race = file.split('_')[:-1]
    ages = np.append(ages, float(age))
    genders = np.append(genders, float(gender))
    races = np.append(races, int(race))
target = np.array([ages, genders, races])
target_train = target[:,:14226]
target_validation = target[:,14226:18966]
target_test = target[:,18966:]
np.save('target_train.npy', target_train)
np.save('target_validation.npy', target_validation)
np.save('target_test.npy', target_test)
