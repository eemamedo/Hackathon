from flask import Flask
app = Flask(__name__)

@app.route('/')
def root():
    return "API is running"

@app.route('/model')
def model():
    return "Model result"

if __name__ == '__main__':
    app.run()

