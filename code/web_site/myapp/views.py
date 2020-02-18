#python manage.py runserver
import subprocess
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    return render(request,'home.html')


def predict(request):
    fever = request.GET["fever"]
    vomiting = request.GET["vomiting"]
    nausea = request.GET["nausea"]
    vomiting_blood = request.GET["vomiting_blood"]
    body_pains = request.GET["body_pains"]
    pain_behind_eyes = request.GET["pain_behind_eyes"]
    joint_pains = request.GET["joint_pains"]
    chill = request.GET["chill"]
    headache = request.GET["headache"]
    swollen_glands = request.GET["swollen_glands"]
    rashes = request.GET["rashes"]
    abdominal_pain = request.GET["abdominal_pain"]
    ble_nose = request.GET["ble_nose"]
    ble_mouth = request.GET["ble_mouth"]
    fatigue = request.GET["fatigue"]
    red_eyes = request.GET["red_eyes"]
    platelets_count = request.GET["platelets_count"]
    values = str(fever)+','+str(vomiting)+','+str(nausea)+','+str(vomiting_blood)+','+str(body_pains)+','+str(pain_behind_eyes)+','+str(joint_pains)+','+str(chill)+','+str(headache)+','+str(swollen_glands)+','+str(rashes)+','+str(abdominal_pain)+','+str(ble_nose)+','+str(ble_mouth)+','+str(fatigue)+','+str(red_eyes)+','+str(platelets_count)
    output = script_function(values)
    lines = str(output.stdout.splitlines())
    # result_text = lines.strip('[b"the recieved values are:').strip(']"]').split(', ')
    # result_text = str(result_text[1].strip(']"'))
    prediction_result = lines.strip("[b'").strip("']")
    if prediction_result == 'True':
        prediction_result = '<h1 style="text-align: center; color: red;">Patient infected with Dengue virus</h1>'
    elif prediction_result == 'False':
        prediction_result = '<h1 style="text-align: center; color: green;">Patient does not have Dengue</h1>'
    else:
        prediction_result = "Unknown"
    return HttpResponse(prediction_result)

def script_function(values):
  return subprocess.run(['python', 'C:/Users/tejak/Desktop/Project@ML/github/code/predict.py', values],stdout=subprocess.PIPE)  