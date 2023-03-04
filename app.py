#importing required modules
import tkinter
import customtkinter
from PIL import ImageTk,Image
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green


app = customtkinter.CTk()  #creating cutstom tkinter window
app.geometry("600x440")
app.title('Kyphosis Diagnosis')

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


img1=ImageTk.PhotoImage(Image.open("images/med.jpg"))
l1=customtkinter.CTkLabel(master=app,image=img1)
l1.pack()

#creating custom frame
frame=customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

l2=customtkinter.CTkLabel(master=frame, text="Kyphosis Diagnosis",font=('Century Gothic',20))
l2.place(x=50, y=45)

entry1=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Number of months')
entry1.place(x=50, y=110)

entry2=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Number of vertebrae involved')
entry2.place(x=50, y=165)

entry3=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='The first vertebrae operated on')
entry3.place(x=50, y=220)

# Prediction function

def predict_survival():
    age = entry1.get()
    number = entry2.get()
    start = entry3.get()
    if age != '' and number != '' and start != '':
        age = float(entry1.get())
        number = int(entry2.get())
        start = int(entry3.get())
        features = [[age, number, start]]
        prediction = rfc.predict(features)
        if prediction[0] == 0:
            messagebox.showinfo("Prediction", "The person is predicted to not have kyphosis.")
        elif prediction[0] == 1:
            messagebox.showinfo("Prediction", "The person is predicted to have kyphosis.")
    elif age == '' or number == '' or start == '':
        result_label.configure(text='Please fill in all the fields.')
#Create custom button
button1 = customtkinter.CTkButton(master=frame, width=220, text="Predict kyphosis", command=predict_survival, corner_radius=6)
button1.place(x=50, y=260)

result_label = customtkinter.CTkLabel(frame, text='Prediction will be shown here.')
result_label.place(x=50, y=300)


app.mainloop()