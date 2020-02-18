import pandas as pd
import sys
#import seaborn as sns
#ignoring feature warnings
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("C:/Users/sanab/Documents/clg_pro_dengu_detection/code/dataset.csv")
data.drop(columns={'id'},inplace=True) #Deleting id
get={'yes':1,'medium':1,'no':0,'high':2,'low':0} 
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
## to split the data into ratio of 75% and 25% to train model and test the model

from sklearn.model_selection import train_test_split

new_data=data.drop(columns={'dengue'})
X_train,X_test,y_train,y_test=train_test_split(new_data,data['dengue'],random_state=7)
#The data is preprocessing using sklearn.preprocessing.Normalizer
from sklearn.preprocessing import Normalizer

norm=Normalizer()
X_train_normal=norm.transform(X_train)
X_test_normal=norm.transform(X_test)
#print(X_test_normal)
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
# from tqdm import tqdm
# results=[]
# names=[]
# scoring='accuracy' 
# for name, model, params in tqdm(models):
#     kfold = KFold(len(X_train_normal), random_state=7, shuffle=True)
#     model_grid = GridSearchCV(model, params)
#     cv_results = cross_val_score(model_grid, X_train_normal, y_train, cv = kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())
#     #print(msg)
#The accuracy score obtained without using GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score, fbeta_score

clf=RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_normal,y_train)
pred=clf.predict(X_test_normal)
accuracy_score(pred,y_test)
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score

clf = DecisionTreeClassifier(random_state=42)

# TODO: Create the parameters list you wish to tune
parameters = {'criterion':['entropy'], 'max_depth':[10, 50, 100]}

# TODO: Make an fbeta_score scoring object


# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf,parameters,scoring='accuracy')

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train_normal,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
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
#prediction function calling
#Enter the values in the sequence fever, vomiting, nausea, vomiting_blood, body_pains, pain_behind_eyes, joint_pains, chill, headache, swollen_glands, rashes, abdominal_pain, ble_nose, ble_mouth, fatigue, red_eyes, platelets_count
recieved_values = str(sys.argv)

print(type(recieved_values))
#print(Predict(100,1,0,1,2,1,0,1,1,1,0,1,1,0,1,0,17000))