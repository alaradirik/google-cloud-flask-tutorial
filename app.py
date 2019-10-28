from flask import Flask, jsonify, request
from yolo import get_predictions


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    predictions = get_predictions(request)

    return jsonify(predictions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)