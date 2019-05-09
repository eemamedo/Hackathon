from flask import Flask, jsonify, render_template
from flask_cors import CORS
import mongo.mongoWrapper as mw
app = Flask(__name__)
CORS(app)

# SCR_PATH = os.path.join('docs', 'SCR')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = SCR_PATH

# @app.route('/')
# @app.route('/index')
# def show_index():
#     full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
#     return render_template("index.html", user_image = full_filename)

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
    return render_template('scr.html', name='scr')


if __name__ == '__main__':
    app.run()

