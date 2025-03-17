# DL-Assignment-1

Please refer to Assignment-1.ipynb for the detailed code of this assignment.

Name: Anurag Bantu
Roll no: MA24M003

# Fashion MNIST Classification

## Project Overview
This project focuses on classifying images from the Fashion MNIST dataset using deep learning. The dataset consists of grayscale images of 10 different clothing categories. The neural networks are created from scratch mostly using only numpy for all matrix/vector operations without using any automatic differentiation packages and are trained on labeled image data to recognize clothing items accurately.

## Dataset
The Fashion MNIST dataset is an alternative to the classic MNIST dataset and contains:
- 60,000 training images
- 10,000 testing images
- 10 classes representing different types of clothing items
- Each image is 28x28 pixels in grayscale

## Requirements
Ensure you have the necessary dependencies installed before running the notebook, including NumPy, Pandas, Matplotlib, and Seaborn.

## Usage
### 1. Load and Preprocess Data
The dataset is loaded and preprocessed by normalizing pixel values and reshaping the images to fit the input format expected by the model.

### 2. Define layer class
Create a layer class that includes all the optimization techniques, the forward propagation and backward propagation algorithms. The following optimization techniques are used:
- SGD – Stochastic Gradient Descent
- Momentum – SGD with Momentum
- Nesterov – Nesterov Accelerated Gradient (NAG)
- RMSprop – Root Mean Square Propagation
- Adam – Adaptive Moment Estimation
- Nadam – Nesterov-accelerated Adaptive Moment Estimation

### 3. Configure wandb and implement the train network function
Set all the possible values for the different parameters to be tested as shown below and the method to be used for the sweep search. 

The parameter values to be tested:

- Epochs: 5, 10, 15
- Number of Hidden Layers: 3, 4, 5
- Fully Connected Layer Size: 32, 64, 128
- Weight Decay: 0, 0.0005, 0.005, 0.05, 0.5
- Learning Rate: 0.001, 0.0001, 0.00001
- Optimizer: SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
- Batch Size: 16, 32, 64
- Weight Initialization: Random, Xavier
- Activation Function: Sigmoid, Tanh, ReLU

Define the train_network function that first creates the neural network with the set configurations and then implements the training through forward and backword propagation. Evaluate the performance of the model during training (train and validation accuracy) and then after training (test accuracy). Cross entropy loss is also calculated at each step.

### 4. Evaluate the best performing Model
The best performing model that gives the highest test accuracy from the wandb sweep is further analyzed using predictions on test dataset to create a confusion matrix.

### 5. Change the loss function
We now use squared error loss instead of cross entropy loss to do similar sweep analysis using wandb to get the best performing models and accuracies with this loss metric.

### 6. Recommendations for MNIST dataset
We select 3 of the best performing and varied models to perform classification on another dataset which is the MNIST (Not Fashion MNIST) dataset. We analyze the performance of these models on this dataset.

For detailed analysis of the performance of the models, refer to the wandb report: https://wandb.ai/ma24m003-iit-madras/neural-network-hyperparam-tuning/reports/Anurag-Bantu-s-DA6401-Assignment-1--VmlldzoxMTc3OTExMQ
