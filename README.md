## Introduction

Accompanying code for the **Deploying a Custom ML Prediction Service on Google Cloud** tutorial on Medium.

## Project Structure

    .
    ├── yolo-coco                       # Pretrained model files
        └── coco.names                  # Available object labels
        └── yolov3.cfg                  # YOLOv3 model configuration
        └── yolov3.weights              # YOLOv3 model weights              
    ├── app.py                          # Flask app serving predictions
    ├── yolo.py                         # Functions to load YOLO and generate predictions
    ├── requirements.txt                # Dependencies
    └── README.md


## Installation Instructions

#### Create your python environment. If using conda:

`conda create -n [name of enviroment] python=3.7`

#### Installation of dependencies:

`pip install -r requirements.txt`


## Running the Flask app on your local:

`cd model-trainer`

`python app.py`

Edit and use `request.py` by specifying the image to be annotated. 
