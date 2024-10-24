# Deep-Learning-for-Eyes-Disease-Detection
This project uses a lightweight and efficient CNN model, SqueezeNet, to detect eye diseases, specifically Pathologic Myopia (PM). SqueezeNet is known for its small size, with fewer parameters compared to AlexNet, while maintaining similar performance.

Project Overview
The main goal of this project is to use deep learning techniques to classify retinal images from the iChallenge-PM dataset into two categories: Pathologic Myopia (PM) and Non-Pathologic Myopia.

Table of Contents
Environment Setup
Dataset
Dataset Structure
Data Preprocessing
Model
Building the Model
Training
Result Visualization
Individual Image Prediction
Environment Setup
The project is developed with the following environment:

Python: 3.7
IDE: PyCharm
Deep Learning Framework: PyTorch
Dataset: iChallenge-PM (Pathologic Myopia Detection)
To get the dataset, visit the dataset link (Extraction code: 2357).

Dataset
1.1 Dataset Introduction
The dataset used in this project is from the iChallenge-PM competition hosted by Baidu Brain and Zhongshan Ophthalmic Center, Sun Yat-sen University. It contains retinal images of 1200 subjects, split into training, validation, and testing sets, with 400 images each.

1.2 Dataset Structure
The dataset consists of three compressed files:

training.zip: Contains the training images and labels
PALM-Training400.zip: Includes 400 retinal images
PALM-Training400-Annotation-D&F.zip: Annotation file for diagnosis and lesions
validation.zip: Contains validation images
valid_gt.zip: Contains the labels for the validation images
PM_Lable_and_Fovea_Location.xlsx: Label and fovea location information
