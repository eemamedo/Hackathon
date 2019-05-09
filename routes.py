from flask import Flask, jsonify
from flask_cors import CORS
import mongo.mongoWrapper as mw
app = Flask(__name__)
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
    for doc in col:
        powerList.append({'x':doc['Time'], 'y': doc['Power']})
        currentList.append({'x':doc['Time'], 'y': doc['Current']})

    payload = {}
    payload['powerList'] = powerList
    payload['currentList'] = currentList
        
    return jsonify(payload)


if __name__ == '__main__':
    app.run()

