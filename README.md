# Udacity AWS Machine Learning Engineer Nanodegree Capstone Project
# American Sign Language hand gesture recognition using CNN Transfer learning

## Project Overview:
* Sign languages are an important means of communication for individuals with hearing impairments or speech disabilities. However, access to sign language education and resources can be limited, especially for people with special needs. This project aims to use computer vision technology to make sign language education more accessible and inclusive for people with special needs.
* The project focuses on developing a Convolutional Neural Network (CNN) model for recognizing sign language gestures. The model will be trained using a pre-trained model as a base, which will be fine-tuned to better recognize the specific sign language being used. The use of a pre-trained model will minimize the amount of data required to train the model and make the process more efficient.
* The developed model will be made available online for individuals with special needs to use for learning sign language. The user-friendly interface and real-time gesture recognition will provide an interactive and engaging learning experience, making sign * language education more accessible and inclusive.
This project has the potential to improve the lives of individuals with special needs by empowering them to communicate effectively using sign language. By leveraging cutting-edge technology, this project aims to bridge the gap in access to sign language education and promote inclusivity for people with special needs.

## Problem Statement:
* Individuals who are specially enabled often face difficulties in accessing resources for learning sign languages, particularly American Sign Language (ASL) alphabets. This can pose challenges to their ability to communicate effectively and limit their full participation in society.
* The aim of this project is to develop a computer vision model that can accurately recognize ASL alphabets, A through Z, to make sign language education more accessible to these individuals. The model will be based on a pre-trained ResNet architecture and fine-tuned to recognize the 29 classes of ASL alphabets. The challenge is to fine-tune the pre-trained model to accurately recognize the subtle differences between ASL alphabets and minimize the error rate in their recognition.
* The goal is to create a model with high accuracy and minimal error in recognizing ASL alphabets, which can be made available online for individuals who are specially enabled to use for learning sign language. This model has the potential to play a crucial role in promoting inclusivity and access to sign language education for all.

# Dataset:
Data is collected from Kaggle https://www.kaggle.com/datasets/shadman0786/asl-alphabet-cnn . It has two folders – Train and Test, each folder has 29 classes – A through Z, Del, Space, nothing. Each training class has 2400 images, and each testing class has 600 images.
<img width="683" alt="image" src="https://user-images.githubusercontent.com/121497007/218170201-493db03b-32e1-47a3-bac2-1259f91516a7.png">

Data distribution:

* Test <br>
-- A (600) <br>
-- B (600) <br>
   . <br>
   . <br>
   . <br>
-- Z (600)

* Train <br>
-- A (2400) <br>
-- B (2400) <br>
   . <br>
   . <br>
   . <br>
-- Z (2400)
   
# Proposal Review is provided in the documents attached

Libraries used in the project:
1- Pytorch - Used for training deep learning ResNet-50 Model
2- shutil - used for copying files
3- Boto3 - For Endpoint invokations
4- Standard libraries for ML and DL processes.

# Project setup and installation

This project can be performed in any of the three softwares: AWS Sagemaker Studio, Jupyter Lab/ Notebooks, or Google Colab. Open the "train_and_deploy.ipynb" file and start by installing all the dependencies. For ease of use you may want to use a Kernel with GPU so that the training process is quick and time saving.

Dataset required is provided in the dataset section
![s3 train folder](https://user-images.githubusercontent.com/121497007/218173744-8c6a8265-635c-4ae1-ad7a-e5cabdec2310.jpg)

# Overview of Project steps:
* We have used data from Kaggle, link is mentioned below: https://www.kaggle.com/datasets/shadman0786/asl-alphabet-cnn
(https://www.kaggle.com/datasets/shadman0786/asl-alphabet-cnn)
* We will be using a pretrained Resnet50 model from pytorch vision library
(https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
(https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html))
* We will add two Fully connected Neural Network layers on top of the above Resnet50 model
We will use concept of Transfer learning therefore we will be freezing all the existing Convolutional layers in the pretrained Resnet50 model and only
change the gradients for the two fully connected layers
* We perform Hyperparameter tuning, to get the optimal best hyperparameters to be used in our model
* We have added configuration for Profiling and Debugging our training model by adding relevant hooks in Training and Testing (evel) phases
* We will then deploy our Model, for deploying we have created inference script. The inference script will be overriding a few functions that will be used
by our deployed endpoint for making inferences/predictions.

# Files used:
* hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test the models with different hyperparameters to find the best hyperparameter
* train_model.py - This script file contains the code that will be used by the training job to train and test the model with the best hyperparameters that we got from hyperparameter tuning
* endpoint_inference.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.
* train_and_deploy.ipynb -- This jupyter notebook contains all the code and steps that we performed in this project and their outputs.

# Hyperparameter Tuning
* The Resnet50 Model with two fully connected Neural network layers are used for the image classification problem. Resnet-50 is 50 layers deep NN and is trained on million images of 1000 categories from the ImageNet Database.

* The optimizer that we will be using for this model is AdamW ( For more info refer : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )

* Hence, the hyperparameters selected for tuning were:

# Endpoint:

-- Learning rate - default(x) is 0.001 , so we have selected 0.01x to 100x range for the learing rate

-- eps - defaut is 1e-08 , which is acceptable in most cases so we have selected a range of 1e-09 to 1e-08

-- Weight decay - default(x) is 0.01 , so we have selected 0.1x to 10x range for the weight decay

-- Batch size -- selected only two values [ 64, 128 ]

Evaluation:
Training and testing accuracy: 60%
model is trained using hyperparameter tuning, using evaluation: accuracy loss.

# HyperParameter Tuning job pic:
![Hypertuning job](https://user-images.githubusercontent.com/121497007/218173184-9f3213e7-d730-48ae-8512-d6a0f2dbb0e4.jpg)

## Multiple training jobs triggered by the HyperParameter tuning job pic:
![Multiple hyperparameter training jobs](https://user-images.githubusercontent.com/121497007/218173068-01b37aba-d239-4a14-963a-29eca7b03196.jpg)

# Endpoint:
Active endpoint:
![Endpoint](https://user-images.githubusercontent.com/121497007/218173594-43355ef8-e73b-44d6-b2b3-c85092126a79.jpg)

After deploying endpoint, I created a lambda function and created asynchronouus trigger, so whenever a input test file is upload in the below location:
![S3-bucket-lambda_s3-invocations](https://user-images.githubusercontent.com/121497007/218174205-c16bf048-74b4-41b0-aa79-b2d81acc3698.jpg)

then Endpoint is triggered using lambda handler, and endpoint is triggered using 'predictor' python file in the lambda function

Lambda function is shown below:
![lambda-function](https://user-images.githubusercontent.com/121497007/218174336-8af1cb4b-a7df-4ad7-8ff4-95a49e913ede.jpg)

Lambda's predictor .py file is shown below:
![predictor-function](https://user-images.githubusercontent.com/121497007/218174592-3b0906ca-1169-4757-8c69-28121ce77b2c.jpg)

Lambda invocation using asynchronous triggering:
![Lambda-endpoint-invocation](https://user-images.githubusercontent.com/121497007/218174726-8b46af07-4f8c-410d-b7c6-88704c95ecac.jpg)

Future scope:
* Implementation of the model and creating an app using RestAPI to create real time predictions, I have got training and testing accuracy around 60%, but that can be increased by increasing the number of testing data images.

Below are some of the predictions and visualized them:
<img width="524" alt="image" src="https://user-images.githubusercontent.com/121497007/218175243-a34150c9-3f24-490a-bbb5-e21512af9b0e.png">
