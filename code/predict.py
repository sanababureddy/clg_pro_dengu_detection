import pandas as pd
import sys
#ignoring feature warnings
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("C:/Users/tejak/Desktop/Project@ML/github/code/dataset.csv")
#data=pd.read_csv("C:/Users/sanab/Documents/clg_pro_dengu_detection/code/dataset.csv")
data.drop(columns={'id'},inplace=True) #Deleting id
get={'no':0,'yes':1,'low':2,'medium':3,'high':4} 
data.vomiting=data.vomiting.map(get)
data.nausea=data.nausea.map(get)
data.vomiting_blood=data.vomiting_blood.map(get)
data.body_pains=data.body_pains.map(get)
data.pain_behind_eyes=data.pain_behind_eyes.map(get)
data.joint_pains=data.joint_pains.map(get)
data.chill=data.chill.map(get)
data.headache=data.headache.map(get)
data.swollen_glands=data.swollen_glands.map(get)
data.rashes=data.rashes.map(get)
data.abdominal_pain=data.abdominal_pain.map(get)
data.ble_nose=data.ble_nose.map(get)
data.ble_mouth=data.ble_mouth.map(get)
data.fatigue=data.fatigue.map(get)
data.red_eyes=data.red_eyes.map(get)
data.dengue=data.dengue.map(get)
data=data[:225] #using 225 rows
from sklearn.model_selection import train_test_split
new_data=data.drop(columns={'dengue'})
X_train,X_test,y_train,y_test=train_test_split(new_data,data['dengue'],random_state=7)
from sklearn.preprocessing import Normalizer
norm=Normalizer()
X_train_normal=norm.transform(X_train)
X_test_normal=norm.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
RF_params = {'n_estimators':[10,50,100]}
DTC_params = {'criterion':['entropy'], 'max_depth':[10, 50, 100]}
LR_params = {'C':[0.001, 0.1, 1, 10, 100]}
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
models=[]
models.append(('DTC', DecisionTreeClassifier(), DTC_params))
models.append(('LR', LogisticRegression(), LR_params))
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
clf=RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_normal,y_train)
pred=clf.predict(X_test_normal)
accuracy_score(pred,y_test)
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
clf = DecisionTreeClassifier(random_state=42)
parameters = {'criterion':['entropy'], 'max_depth':[10, 50, 100]}
grid_obj = GridSearchCV(clf,parameters,scoring='accuracy')
grid_fit = grid_obj.fit(X_train_normal,y_train)
best_clf = grid_fit.best_estimator_
predictions = (clf.fit(X_train_normal, y_train)).predict(X_test_normal)
best_predictions = best_clf.predict(X_test_normal)
def Predict(fever, vomiting, nausea, vomiting_blood, body_pains, pain_behind_eyes, joint_pains, chill, headache, swollen_glands, rashes, abdominal_pain, ble_nose, ble_mouth, fatigue, red_eyes, platelets_count):
    normilized_value = norm.transform([[fever, vomiting, nausea, vomiting_blood, body_pains, pain_behind_eyes, joint_pains, chill, headache, swollen_glands, rashes, abdominal_pain, ble_nose, ble_mouth, fatigue, red_eyes, platelets_count]]);
    element = str(normilized_value[[0]])
    element = element.strip('[').strip(']').split()
    prediction_result=best_clf.predict([element])
    if prediction_result == [1]:
        return True
    elif prediction_result == [0]:
        return False
    else:
        return "Unknown"
recieved_values = str(sys.argv)
rv = recieved_values.strip('[').strip(']').strip("'").strip(" '").split(',')
fever = float(rv[1].strip(" '"))
vomiting = int(rv[2])
nausea =  int(rv[3])
vomiting_blood =  int(rv[4])
body_pains =  int(rv[5])
pain_behind_eyes =  int(rv[6])
joint_pains =  int(rv[7])
chill =  int(rv[8])
headache =  int(rv[9])
swollen_glands =  int(rv[10])
rashes =  int(rv[11])
abdominal_pain =  int(rv[12])
ble_nose =  int(rv[13])
ble_mouth =  int(rv[14])
fatigue =  int(rv[15])
red_eyes =  int(rv[16])
platelets_count =  int(rv[17])
print(Predict(fever, vomiting, nausea, vomiting_blood, body_pains, pain_behind_eyes, joint_pains, chill, headache, swollen_glands, rashes, abdominal_pain, ble_nose, ble_mouth, fatigue, red_eyes, platelets_count))
