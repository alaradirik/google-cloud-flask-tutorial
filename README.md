## Introduction

Accompanying code for the **Deploying a ML Model on Google Compute Engine** tutorial on Medium.

## Project Structure

    .
    ├── yolo-coco                       # Pre-trained model files
        └── coco.names                  # Object labels (person, car, etc.)
        └── yolov3.cfg                  # Model configuration
        └── yolov3.weights              # Model weights              
    ├── app.py                          # Flask app serving predictions
    ├── yolo.py                         # Functions to generate predictions
    ├── requirements.txt                # Dependencies
    └── README.md


## Installation Instructions

#### Create your python environment. If using conda:

`conda create -n [name of enviroment] python=3.7`

#### Installation of dependencies:

`pip install -r requirements.txt`


## Running the Flask app on your local:

`cd google-cloud-flask-tutorial`

`conda activate [name of enviroment]`

`python app.py`

Edit and use `request.py` to send a POST request to the app. 
