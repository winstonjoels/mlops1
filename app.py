from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pickle
import numpy as np
import random
import csv
from sklearn.metrics import accuracy_score
import pandas as pd
import github3
import os

def randN():
    N=7
    min = pow(10, N-1)
    max = pow(10, N) - 1
    id = random.randint(min, max)
    return id

app = Flask(__name__)

global model, cols, id

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/eval',methods=['POST'])
def eval():
    filename = "data/Details.csv"
    fields = [] 
    rows = [] 
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    df = pd.DataFrame(rows, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    df = df.loc[df['Actual'] == '?']
    return render_template("eval.html", column_names=df.columns.values, row_data=list(df.values.tolist()), link_column="Actual", zip=zip)

@app.route('/after_store',methods=['POST'])
def after_eval():
    columns1 = ['ID', 'Name', 'Predicted', 'Actual']
    int_features = [x for x in request.form.values()]
    print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = columns1)
    
    output = []
    filename = "data/Details.csv"
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for line in csvreader:
            if str(data_unseen['ID'][0]) == line[0]:
                line[3] = str(data_unseen['Actual'][0])
            output.append(line)
    list2 = ['ID','Name', 'Predicted', 'Actual']
    df = pd.DataFrame(output, columns=list2)
    df.to_csv('data/Details.csv', mode='w', header=False, index=False)

    # files_to_upload = 'data/Details.csv'
    # username='Juggernaut1997'
    # gh = github3.login(token='d5e1ad22339d9eaed89b794e0d47266ca43cc313')
    # repository = gh.repository(username, 'churn_prediction_gcp')
    # file_size = os.stat(files_to_upload).st_size
    # if file_size !=0:
    #     with open(files_to_upload, 'rb') as fd:
    #         contents = fd.read()
    #     contents_object = repository.file_contents(files_to_upload)
    #     contents_object.update(df.to_csv(header=False), contents)
    # else:
    #     pass
    return home()

@app.route('/previous_data_list',methods=['POST'])
def prev_data_list():
    x = np.arange(10)
    fig = go.Figure(data=go.Scatter(x=x, y=x ** 2))
    fig.write_html("graph.html")
    return render_template("choose_model.html")
    
@app.route('/previous_data/<name>',methods=['POST'])
def prev_data(name):
    filename = "data/Details.csv"
    fields = [] 
    rows = [] 
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row)
    df = pd.DataFrame(rows, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    df = df.loc[df['Name'] == name]
    return render_template("view_data.html", name=name, data=df.to_html(index=False))

@app.route('/model/<name>',methods=['GET','POST'])
def model(name):
    global model, cols, id
    id = randN()
    if name=="infy_bank":
        cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    else:
        cols = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
    file = name+".html"
    return render_template(file, id=id)


@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model, cols, id, predict
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    if name == 'mush':
        name1 = name+"_training_pipeline"
    else:
        name1 = name
    model = load_model(name1)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    pred = int(prediction)

    actual = '?'
    list = [id, name, int(prediction), actual]
    list2 = ['ID', 'Name', 'Predicted', 'Actual']
    df = np.array(list)
    df = pd.DataFrame([df], columns=list2)

    df.to_csv('data/Details.csv', mode='a', header=False, index=False)

    file = name + ".html"
    return render_template(file, id=id, name=name, pred=pred)

@app.route('/model_eval',methods=['GET','POST'])
def model_eval():
    global accu
    filename = "data/Details.csv"
    name = ['infy_bank', 'mush']
    rows = []

    # with open(filename, 'r') as csvfile:
    #     csvreader = csv.reader(csvfile)
    #     for row in csvreader:
    #         rows.append(row)
    #
    url = 'https://raw.githubusercontent.com/Juggernaut1997/churn_prediction_gcp/master/data/Details.csv'
    df = pd.read_csv(url)
    # df = np.array(rows)
    # df = pd.DataFrame(df, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    accuracy = {}
    for n in name:
        acc = "Not_enough_data"
        df1 = df.loc[df['Name'] == n].loc[df['Actual'] != '?']
        points = str(len(df1['Name'].to_list()))
        y_true = df1['Actual'].to_list()
        y_pred = df1['Predicted'].to_list()
        if len(y_true)>=2 and len(y_pred)>=2:
            acc = str(accuracy_score(y_true, y_pred)*100)
        list1 = [n, points, acc]
        list1 = np.array(list1)
        list3 = ['Name','Data Points','Accuracy']
        df2 = pd.DataFrame([list1], columns=list3)
        print(df2)
        df2.to_csv('data/accuracy.csv', mode='a', header=False, index=False)
        
    return render_template("model_eval.html", accuracy = accuracy, name = name)
