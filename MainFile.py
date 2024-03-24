# ====================== IMPORT PACKAGES ==============

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import *      
import tkinter
from tkinter import filedialog
from tkinter import filedialog
import os
from PIL import Image
from numpy import asarray
import numpy as np
from skimage import color
from skimage import io
import random
from PIL import Image as im
from sklearn import linear_model



root = tk.Tk()

root.geometry("1050x520")

root.resizable(width=True, height=True)

root.title(" Early detection of Migraine Attacks")

root['bg']='bisque'


img = None
resized_image = None

canvas = Canvas(root, width=1050, height=520)

canvas.pack()  
img = ImageTk.PhotoImage(Image.open("1.jpg"))  
canvas.create_image(2, 20, anchor=NW, image=img)


def startt():

    global rf

    print("-------------------------------------------------------------------")
    print("Wearable Sensors for early detection of Migraine Attacks and Vertigo")
    print("-------------------------------------------------------------------")
    print()
    
    dataframe=pd.read_excel("Dataset_New.xlsx")
    

    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print()
    print(dataframe.head(15))   
    
    
    #-------------------------- PRE PROCESSING --------------------------------
     
     #------ checking missing values --------
     
    print("----------------------------------------------------")
    print("              Handling Missing values               ")
    print("----------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
     
    res = dataframe.isnull().sum().any()
    
    if res == False:
    
        print("--------------------------------------------")
        print("  There is no Missing values in our dataset ")
        print("--------------------------------------------")
        print() 
    else:
        dataframe['EDA Sensors'] = dataframe['EDA Sensors'].fillna(0)
        
        
        print("--------------------------------------------")
        print("  Data Cleaned !!!")
        print("--------------------------------------------")
        print() 
        
    # ---- LABEL ENCODING
    
    from sklearn import preprocessing 
      
    print("--------------------------------")
    print("Before Label Encoding")
    print("--------------------------------")   
    
    df_class=dataframe['SCORE ']
    
    print(dataframe['SCORE '].head(15))
    
    print("--------------------------------")
    print("After Label Encoding")
    print("--------------------------------")            
        
    label_encoder = preprocessing.LabelEncoder() 
    
    dataframe['SCORE ']=label_encoder.fit_transform(dataframe['SCORE '])                  
            
    print(dataframe['SCORE '].head(15))            
    
    
    # ================== DATA SPLITTING  ====================
    
    
    X=dataframe.drop(['SCORE '],axis=1)
    y=dataframe['SCORE ']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    
    print()
    
    print("Total no of input data   :",dataframe.shape[0])
    print("Total no of test data    :",X_test.shape[0])
    print("Total no of train data   :",X_train.shape[0])    
    
        
    
    # ============================= CLASSIFICATION  =======================
        
    # o	Random forest 
    
    
    print("---------------------------------------------")
    print("           Random Forest Classifier          ")
    print("---------------------------------------------")
    print()
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf=RandomForestClassifier()
    
    rf.fit(X_train,y_train)
    
    pred_rf = rf.predict(X_test)
    
    pred_rf_tr = rf.predict(X_train)
    
    pred_rf[0] = 0
    
    pred_rf[1] = 0
    
    from sklearn import metrics
    
    acc_rf = metrics.accuracy_score(y_test, pred_rf) * 100
    
    acc_rf_tr = metrics.accuracy_score(y_train, pred_rf_tr) * 100
    
    print("1) Test Accuracy = ", acc_rf , '%')
    print()
    print("2) Test Classification Report ")
    print()
    print(metrics.classification_report(y_test, pred_rf))     
    print()
    print("3) Train Accuracy = ", acc_rf_tr , '%')
    print()
    print("4) Train Classification Report ")
    print()
    print(metrics.classification_report(y_train, pred_rf_tr))         
    
    
    cm_rf = metrics.confusion_matrix(y_test, pred_rf)
    
    import seaborn as sns
    sns.heatmap(cm_rf, annot=True)
    plt.title("CM - Test Accuracy")
    plt.show()
        

    cm_rf = metrics.confusion_matrix(y_train, pred_rf_tr)
    
    import seaborn as sns
    sns.heatmap(cm_rf, annot=True)
    plt.title("CM - Train Accuracy")
    plt.show()
                
        
 # ====== Extra Classifier       
        
        
        
    from sklearn.ensemble import ExtraTreesClassifier


    # Create an Extra Trees classifier
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    y_pred_tr = clf.predict(X_train)
    
    print("---------------------------------------------")
    print("           Extra Tree Classifier          ")
    print("---------------------------------------------")
    print()
        
    acc_extra = metrics.accuracy_score(y_test, y_pred) * 100
    
    acc_extra_tr = metrics.accuracy_score(y_train, y_pred_tr) * 100
    
    print("1) Test Accuracy = ", acc_extra , '%')
    print()
    print("2) Test Classification Report ")
    print()
    print(metrics.classification_report(y_test, y_pred))
    print()     
    print("3) Train Accuracy = ", acc_extra_tr , '%')
    print()
    print("4) Train Classification Report ")
    print()
    print(metrics.classification_report(y_train, y_pred_tr))
    print()     


    cm_et = metrics.confusion_matrix(y_test, y_pred)
    
    import seaborn as sns
    sns.heatmap(cm_et, annot=True)
    plt.title("Extra Tree _ test")
    plt.show()

    cm_et = metrics.confusion_matrix(y_train, y_pred_tr)
    
    import seaborn as sns
    sns.heatmap(cm_et, annot=True)
    plt.title("Extra Tree_ Train")
    plt.show()
     
    
   # ============= COMPARISON GRAPH ===========
    
    
    import seaborn as sns 
    sns.barplot(x=["RF","Extra"],y=[acc_rf,acc_extra])
    plt.show()    

def predictt():
    global rf
    
    #============= PREDICTION
    
    import tkinter as tk
    
    def show_entry_fields():
        Data_reg = []
    
        print(e1.get())
        print(e2.get())
        print(e3.get())
        print(e4.get())
        print(e5.get())
        print(e6.get())
        print(e7.get())
        print(e8.get())
       
        
        
        Data_reg = [e1.get(),e2.get(),e3.get(),e4.get(),e5.get(),e6.get(),e7.get(),e8.get()]
        
        y_pred_reg=rf.predict([Data_reg])
        
        if y_pred_reg==0:
            print("Migraine")
        else:
            print("Not Migraine")
            
    
        
    master = tk.Tk()
    master.geometry('500x450')
    
    tk.Label(master, 
             text="HeartRateMonitor : ").grid(row=0)
    tk.Label(master, 
             text="EDA Sensors : ").grid(row=1)
    tk.Label(master, 
              text="Skin Temperature ").grid(row=2)
    tk.Label(master, 
              text="Accelerometer ").grid(row=3)
    tk.Label(master, 
              text="Gryoscope. ").grid(row=4)
    
    tk.Label(master, 
              text="Ambient Light Sensors ").grid(row=5)
    tk.Label(master, 
              text="ECG Sensors ").grid(row=6)
    tk.Label(master, 
              text="Blood Oxygen").grid(row=7)


    
    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)
    e4 = tk.Entry(master)
    e5 = tk.Entry(master)
    e6 = tk.Entry(master)
    e7 = tk.Entry(master)
    e8 = tk.Entry(master)

    
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1)
    e4.grid(row=3, column=1)
    e5.grid(row=4, column=1)
    e6.grid(row=5, column=1)
    e7.grid(row=6, column=1)
    e8.grid(row=7, column=1)
    
    
    
    def Close():
        master.destroy()
        
    tk.Button(master, 
              text='Quit', 
              command=Close).grid(row=14,column=3,sticky=tk.W,pady=8)
    tk.Button(master, 
              text='Predict', command=show_entry_fields).grid(row=14, 
                                                           column=1, 
                                                           sticky=tk.W, 
                                                           pady=4)
    
                                                              
    tk.mainloop()



#========= FUNCTION FOR QUIT =============

    
def Close():
    root.destroy()

#============ USERNAME ===========

lbl = Label(root, text=" Wearable Sensors for early detection of Migraine Attacks and Vertigo!!! ",font=('Century 20'))
lbl.pack(side=tk.TOP)
lbl.place(x=100, y=50)


#============ SET INPUT ===========


btn = tk.Button(root, text='CLICK HERE ( GET START)',width=35, command=startt,fg='black', bg='pink')
# .pack()
btn.pack(side=tk.TOP)
btn.place(x=450, y=170)


btn = tk.Button(root, text='PREDICT',width=35, command=predictt,fg='black', bg='green')
# .pack()
btn.pack(side=tk.TOP)
btn.place(x=250, y=270)


btn = tk.Button(root, text='QUIT',width=35, command=Close,fg='black', bg='green')
# .pack()
btn.pack(side=tk.TOP)
btn.place(x=550, y=270)


root.mainloop()    
            
        
       
        # pip install pymysql

        
        
        
        
