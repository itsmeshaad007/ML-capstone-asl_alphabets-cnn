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

* Test
-- A (600)
-- B (600)
   .
   .
   .
-- Z (600)

* Train
-- A (2400)
-- B (2400)
   .
   .
   .
-- Z (2400)
   
   Proposal Review is provided in the documents attached
   
