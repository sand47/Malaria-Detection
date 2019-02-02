# Malaria-Detection-
Deep learning approach to solve malaria detection using transfer learning in Pytorch <br> 
This project was a part of Udacity Pytorch challenge by Facebook 

## Motivation 
<br>
I was diagnosed with Malaria couple of years ago and by god grace I was cured.  So I decided to build this deep learning model using pytorch which can identify if your diagnosed with malaria or not in no time. Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female.There were an estimated 219 million cases of malaria in 90 countries and deaths rate reached 435000 in 2017. Thus automating the finding will give accurate results to doctors which will help to decrease the death rate. So I started building the model using few layers of CNN but it did not work out well so I tried transfer learning using several architecture and found resnet50 to be best among them. It gave me a 94% accuracy on the validation set.
<br>

## Dataset
You can download the dataset from this link <br>
<br>
https://ceb.nlm.nih.gov/repositories/malaria-datasets/
<br>

## Dependencies
Pytorch (Version 1.0) <br>
<br>
You can download Pytorch from this link https://pytorch.org/ <br>

## Training
<br>
python malaria_detection_train.py 
<br>

## Trained Model
<br>
https://drive.google.com/open?id=10S33Qfz3U8uFdnGL6UBXZqBDWeyKZDbq
<br>

## Testing 

Change the image name in the code and run the following to test single prediction
<br>
python test_single_image.py
<br>



