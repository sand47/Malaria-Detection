# Import needed packages
import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from torchvision import models

def load_checkpoint(checkpoint_path):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    model.eval()
    return model
# Load your model to this variable
model = load_checkpoint('malaria_detector.pt')


def predict_image(image_path):
    print("Prediction in progress")
    image = Image.open(image_path)

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    index = output.data.numpy().argmax()

    return index


if __name__ == "__main__":

    imagefile = "images/nonpara.png"

    imagepath = os.path.join(os.getcwd(), imagefile)
   
    # run prediction function annd obtain prediccted class index
    index = predict_image(imagepath)
    if(index==0):
        print("Postive malaria")
    elif(index==1):
        print("Negative")

   
