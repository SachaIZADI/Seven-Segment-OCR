# Optimizer
## A seven-segment digits OCR

Optimizer is a seven-segment digits OCR class project carried out by [Alex](https://github.com/alexmomeni), [Priscille](https://github.com/priscilleb), [Charlotte](https://github.com/charlottecaucheteux/) and [Sacha](https://github.com/SachaIZADI/).

## Objective
<img src = "img/Pb.png" height="200">
The aim of the project is to digitize the monitoring of mines activities. We focused on the gas and lubricant consumption of vehicles within the mines. The idea is to build computer vision model that would enable operators to take a picture of the gas pump with their smartphones, and automatically log the value of the gas transaction. We were given \~850 pictures (of varying quality) of the gas pump with its associated value.

<img src = "img/product.png" height="200">

## Approaches
<img src = "img/approaches.png" height="200">

We tried 2 different approaches:
1. The "digit-per-digit" approach
    1. Image processing: identify the screen, crop the picture, grayscale, thresholding, localize digits and crop them.
    2. Learning phase: learn a "MNIST" model that predicts each digit individually.
    3. Inference phase: pass each cropped digit to the "MNIST" model, and append the results.

2. The "end-to-end" approach
    1. TBC
<img src = "img/NN.png" height="200">

## Results
<img src = "img/results1.png" height="250">
<img src = "img/Results2.png" height="250">

## Running the models

TBC...



#1 Preprocessing
python frame_extractor.py

#2 Preprocessing
python digits_cut.py

#3 Model
python main.py
