from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask import request
import os
import pandas as pd
import mongo.mongoWrapper as mw

DOCS_PATH = os.path.join('static', 'scrs')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = DOCS_PATH
CORS(app)

@app.route('/')
def root():
    return "API is running"

@app.route('/sensorDataFake')
def sensorDataFake():
    payload = []
    payload.append({
        "Time": "2017-08-23 0:00",
        "Power": 1989,
        "Voltage": 555
    })
    payload.append({
        "Time": "2017-08-23 0:10",
        "Power": 1822,
        "Voltage": 560
    })
    payload.append({
        "Time": "2017-08-23 0:20",
        "Power": 1800,
        "Voltage": 540
    })
    return jsonify(payload)

@app.route('/sensorData')
def sensorData():
    powerList = []
    currentList = []

    f = open("credentials.txt", "r")
    password = f.read()                                                         
    db = mw.getDBRef(password, "FinalTimeData")

    col = mw.queryCol(db, "SensorReadings")
    for i, doc in enumerate(col):
        powerList.append({'x':i, 'y': doc['Power']})
        currentList.append({'x':i, 'y': doc['Current']})

    payload = {}
    payload['powerList'] = powerList
    payload['currentList'] = currentList
        
    return jsonify(payload)

@app.route('/scr')
def scr():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'SCR.jpg')
    return render_template("scr.html", scr_image = full_filename)

@app.route('/faultPredict')
def faultPredict():
    time = request.args.get('time')

    predictionList = []

    index = 302013

    data = pd.read_csv("Results.csv")
    for i, row in data.iterrows():
        predictionList.append({'index':index, 'score': row['r2_score']})
        index -= 1

    return jsonify(predictionList)


if __name__ == '__main__':
    app.run()

 