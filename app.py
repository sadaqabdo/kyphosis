# Import necessary libraries
import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from ttkthemes import ThemedStyle


# Load kyphosis dataset
kyphosis_data = pd.read_csv('kyphosis.csv')

# Preprocess the data
kyphosis_data['Kyphosis'] = kyphosis_data['Kyphosis'].apply(lambda x: 1 if x == 'present' else 0)
X = kyphosis_data.drop(['Kyphosis'], axis=1)
y = kyphosis_data['Kyphosis']
# SMOTE for imbalanced dataset
def balance_dataset(X, y):
    sm = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

X_res, y_res = balance_dataset(X, y) 
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, 
                                                    y_res, 
                                                    test_size=0.2, 
                                                    random_state=42)
# Train the model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Create GUI
root = tk.Tk()
root.title('kyphosis Predictor')

style = ThemedStyle(root)
style.set_theme('smog')

print('root : ',type(root))
age_label = tk.Label(root, text='Enter age:')
age_label.pack()

age_entry = tk.Entry(root)
age_entry.pack()

number_label = tk.Label(root, text='Enter number :')
number_label.pack()

number_entry = tk.Entry(root)
number_entry.pack()

start_label = tk.Label(root, text='Enter start :')
start_label.pack()

start_entry = tk.Entry(root)
start_entry.pack()



def predict_survival():
    age = float(age_entry.get())
    gender = int(number_entry.get())
    pclass = int(start_entry.get())
    features = [[gender, pclass, age]]
    prediction = rfc.predict(features)
    if prediction[0] == 0:
        result_label.config(text='The person is predicted to not have kyphosis.')
    else:
        result_label.config(text='The person is predicted to have kyphosis.')

predict_button = tk.Button(root, text='Predict survival', command=predict_survival)
result_label = tk.Label(root, text='')
result_label.pack()
predict_button.pack()



root.mainloop()
