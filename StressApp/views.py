from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import io
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
from sklearn.preprocessing import Normalizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import FederatedModel
from FederatedServer import FederatedServer #load federated server model
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import pymysql
from datetime import datetime
import shap #loading SHAP tool for XAI based explanation

global username, X, Y, dataset, fs
global uname, vc_cls, scaler, label_encoder
global X_train, X_test, y_train, y_test, labels
global accuracy, precision, recall, fscore, columns

fs = FederatedServer() #creating federated server object
detection_model_path = 'model/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
fs = FederatedServer() #creating federated server object

def PredictVideo(request):
    if request.method == 'GET':
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            emotion_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        emotion_classifier.load_weights("model/cnnmodel_weights.h5")
        cap = cv2.VideoCapture(0)
        while True:
            output = ""
            ret, frame = cap.read()
            if ret == True:
                temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(temp,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
                    roi = frame[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (32, 32))
                    roi = roi.reshape(1,32,32,3)
                    roi = roi.astype('float32')
                    img = roi/255
                    preds = emotion_classifier.predict(img)
                    predict = np.argmax(preds)
                    if predict == 0 or predict == 1 or predict == 2:
                        output = "High Stress Detected"
                    elif predict == 3 or predict == 4:
                        output = "Low Stress Detected"
                    else:
                        output = "Medium Stress Detected"
                    cv2.putText(frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                cv2.imshow("Stress Detection", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()    
        context= {'data':'Video Processing Completed'}
        return render(request, 'UserScreen.html', context)   

def PredictImageAction(request):
    if request.method == 'POST':        
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("StressApp/static/test.jpg"):
            os.remove("StressApp/static/test.jpg")
        with open("StressApp/static/test.jpg", "wb") as file:
            file.write(myfile)
        file.close()
        output = "No Stress detected"
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            emotion_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        emotion_classifier.load_weights("model/cnnmodel_weights.h5")
        orig_frame = cv2.imread('StressApp/static/test.jpg')
        frame = cv2.imread('StressApp/static/test.jpg',0)
        faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        print("==================="+str(len(faces)))   
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = orig_frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (32, 32))
            roi = roi.reshape(1,32,32,3)
            roi = roi.astype('float32')
            img = roi/255
            preds = emotion_classifier.predict(img)
            predict = np.argmax(preds)
            if predict == 0 or predict == 1 or predict == 2:
                output = "High Stress Detected"
            elif predict == 3 or predict == 4:
                output = "Low Stress Detected"
            else:
                output = "Medium Stress Detected"
        cv2.putText(orig_frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(orig_frame)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)                

def ViewWeights(request):
    if request.method == 'GET':
        global fs
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Date</th>'
        output += '<th><font size="" color="black">Model Weight Values</th>'
        output+='</tr>'
        weight, data = fs.getDetails()
        for i in range(len(data)):
            value = data[i]
            output += '<td><font size="" color="black">'+value[0]+'</td><td><font size="" color="black">'+str(value[1])+'</td>'
            output += '<td><font size="" color="black">'+str(weight[i])+'</td></tr>'            
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)

def RunFS(request):
    if request.method == 'GET':
        global vc_cls, fs
        estimate = vc_cls.estimators_
        estimate = estimate[2]
        estimate = estimate.feature_importances_
        estimates = " ".join(str(x) for x in estimate)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        fs.update("admin", str(current_datetime), estimates)
        context= {'data':'ML model state successfully saved at Federated Server'}
        return render(request, 'AdminScreen.html', context)

def Explain(request):
    if request.method == 'GET':
        global vc_cls, fs, X_train, columns
        estimate = vc_cls.estimators_
        estimate = estimate[2]
        explainer = shap.TreeExplainer(estimate, X_test)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        shap.summary_plot(shap_values[0], feature_names=columns, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':"Shap Explanation", 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def PredictImage(request):
    if request.method == 'GET':
        return render(request, 'PredictImage.html', {})

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset, labels, Y, X, label_encoder, scaler, X_train, X_test, y_train, y_test, columns
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("StressApp/static/"+fname):
            os.remove("StressApp/static/"+fname)
        with open("StressApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("Dataset/corporate_stress_dataset.csv", usecols=['Age', 'Gender', 'Working_Hours_per_Week', 'Remote_Work', 'Stress_Level', 'Health_Issues',
                                                                       'Sleep_Hours', 'Physical_Activity_Hours_per_Week', 'Mental_Health_Leave_Taken',
                                                                       'Work_Pressure_Level', 'Annual_Leaves_Taken'])
        columns = dataset.columns
        labels = ['Low', 'Medium', 'High']
        stress = dataset['Stress_Level'].ravel()
        Y = []
        for i in range(len(stress)):
            if stress[i] < 4:
                Y.append(0)
            elif stress[i] >= 4 and stress[i] < 8:
                Y.append(1)
            else:
                Y.append(2)
        Y = np.asarray(Y)        
        datas = dataset.values
        dataset["Remote_Work"] = dataset["Remote_Work"].astype(int)
        dataset["Mental_Health_Leave_Taken"] = dataset["Mental_Health_Leave_Taken"].astype(int)
        label_encoder = []
        columns = dataset.columns
        types = dataset.dtypes.values
        for j in range(len(types)):
            name = types[j]
            if name == 'object': #finding column with object type
                le = LabelEncoder()
                dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
                label_encoder.append([columns[j], le])
        dataset.fillna(0, inplace = True)#replace missing values
        dataset.drop(['Stress_Level'], axis = 1,inplace=True)
        X = dataset.values
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)#shuffle dataset values
        X = X[indices]
        Y = Y[indices]
        X = X[0:3000]
        Y = Y[0:3000]
        Y = np.asarray(Y)
        unique, count = np.unique(Y, return_counts=True)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        data = np.load("model/data.npy", allow_pickle=True)
        X_train, X_test, y_train, y_test = data
        output = "Total records found in dataset = <font size=3 color=blue>"+str(X.shape[0])+"</font><br/>"
        output += "Total features found in Dataset = <font size=3 color=blue>"+str(X.shape[1])+"</font><br/>"
        output += "80% dataset records used to train algorithms = <font size=3 color=blue>"+str(X_train.shape[0])+"</font><br/>"
        output += "20% dataset records used to test algorithms = <font size=3 color=blue>"+str(X_test.shape[0])+"</font><br/><br/>"                                                                                                      
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</font></th>'
        output += '</tr>'
        for i in range(0, 300):
            output += '<tr>'
            for j in range(len(datas[i])):
                output += '<td><font size="3" color="black">'+str(datas[i,j])+'</font></td>'
            output += '</tr>'
        output+= "</table></br>"
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize = (4, 3)) 
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Instances")
        plt.title("Different Class Labels Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def calculateMetrics(algorithm, y_test, predict):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(round(a, 4))
    precision.append(round(p, 4))
    recall.append(round(r, 4))
    fscore.append(round(f, 4))

def PredictAction(request):
    global username
    if request.method == 'POST':
        global vc_cls, labels, scaler, label_encoder
        age = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        working_hours = request.POST.get('t3', False)
        remote_work = request.POST.get('t4', False)
        health_issue = request.POST.get('t5', False)
        sleep_hours = request.POST.get('t6', False)
        physical_activity = request.POST.get('t7', False)
        mental_health = request.POST.get('t8', False)
        work_pressure = request.POST.get('t9', False)
        annual_leave = request.POST.get('t10', False)

        data = []
        data.append([int(age.strip()), gender.strip(), int(working_hours.strip()), int(remote_work.strip()), health_issue.strip(), float(sleep_hours.strip()),
                     float(physical_activity.strip()), int(mental_health.strip()), int(work_pressure.strip()), int(annual_leave.strip())])
        data = pd.DataFrame(data, columns=['Age','Gender','Working_Hours_per_Week','Remote_Work','Health_Issues','Sleep_Hours','Physical_Activity_Hours_per_Week',
                                           'Mental_Health_Leave_Taken','Work_Pressure_Level','Annual_Leaves_Taken'])
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            data[le[0]] = pd.Series(le[1].transform(data[le[0]].astype(str)))#encode all str columns to numeric
        data = data.values
        data = scaler.transform(data)
        predict = vc_cls.predict(data)[0]
        print(predict)
        if predict == 0:
            status = "<font size=3 color=green>Low</font>"
        if predict == 1:
            status = "<font size=3 color=orange>Medium</font"
        if predict == 2:
            status = "<font size=3 color=red>High</font"
        context= {'data':"Predicted Stress = "+status}
        return render(request, 'Predict.html', context)                        

def trainAlgorithms(X_train, X_test, y_train, y_test):
    global vc_cls, labels
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", y_test, predict)
    
    lr_cls = LogisticRegression()
    lr_cls.fit(X_train, y_train)
    predict = lr_cls.predict(X_test)
    calculateMetrics("Logistic Regression", y_test, predict)

    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)
    predict = nb_cls.predict(X_test)
    calculateMetrics("Naive Bayes", y_test, predict)

    xg_cls = XGBClassifier(n_estimators=30)
    xg_cls.fit(X_train, y_train)
    predict = xg_cls.predict(X_test)
    calculateMetrics("XGBoost", y_test, predict)

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    estimators = [('dt', dt), ('rf', rf), ('xg', xg_cls)]
    vc_cls = VotingClassifier(estimators = estimators)
    vc_cls.fit(X_train, y_train)
    predict = vc_cls.predict(X_test)
    calculateMetrics("Ensemble Model", y_test, predict)
    conf_matrix = confusion_matrix(y_test, predict)
                       
    output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
    output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
    output+='</tr>'
    algorithms = ['SVM', 'Logistic Regression', 'Naive Bayes', 'XGBoost', 'Ensemble Model']
    for i in range(len(algorithms)):
        output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
        output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
    output+= "</table></br>"
    df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FSCORE',fscore[0]],
                       ['Logistic Regression','Accuracy',accuracy[1]],['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','FSCORE',fscore[1]],
                       ['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','FSCORE',fscore[2]],
                       ['XGBoost','Accuracy',accuracy[3]],['XGBoost','Precision',precision[3]],['XGBoost','Recall',recall[3]],['XGBoost','FSCORE',fscore[3]],
                       ['Ensemble Model','Accuracy',accuracy[4]],['Ensemble Model','Precision',precision[4]],['Ensemble Model','Recall',recall[4]],['Ensemble Model','FSCORE',fscore[4]],
                     ],columns=['Parameters','Algorithms','Value'])
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
    axis[0].set_title("Confusion Matrix Prediction Graph")
    axis[1].set_title("All Algorithms Performance Graph")
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
    ax.set_ylim([0,len(labels)])    
    df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
    plt.title("All Algorithms Performance Graph")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.clf()
    plt.cla()
    return output, img_b64


def RunML(request):
    if request.method == 'GET':
        global uname, vc_cls, scaler, label_encoder, X, Y, dataset
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore
        accuracy = []
        precision = []
        recall = [] 
        fscore = []
        output, img_b64 = trainAlgorithms(X_train, X_test, y_train, y_test)
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'stress',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'stress',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Stress Detection activities"
        context= {'data':output}
        return render(request, 'Register.html', context)
        

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'stress',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})         

def AdminLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'Welcome '+username}
            return render(request, "AdminScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'AdminLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

