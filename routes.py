from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def root():
    return "API is running"

@app.route('/sensorData')
def sensorData():
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



if __name__ == '__main__':
    app.run()

