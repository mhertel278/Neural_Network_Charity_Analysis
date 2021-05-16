# Neural_Network_Charity_Analysis

## Resources
Python 3.7.9, Scikit-Learn 0.24.1, Tensorflow 2.4.1

## Overview

The purpose of this project was to use build a neural network deep learning model to predict if organizations would likely receive their requested donations from a particular non-profit.  I used a dataset containing metadata about organizations that had previously requested donations, including whether or not their requests were granted.  The goal was to create a model with at least 75% accuracy.

I then preprocessed the data by dropping columns, grouping some unique values in 'Other' bins for certain columns to reduce noisy data, used OneHotEncoder to transform categorical variables into numerical data, divided the variable columns into features and target, split the data into training and testing sets, and finally scaled the features data using StandardScaler.

I then built a Neural Network Deep Learning Model.  After testing the model and determining the initial model structure did not achieve the desired accuracy, I tweaked the model multiple times in attempt to improve the accuracy.  I saved the weight coefficients for each version of the model, and for the model with the highest accuracy I saved the entire model.

## Reults

- Data Preprocessing

    - The target variable was the 'IS_SUCCESSFUL' column from the original data set, as the goal of the model was to predict if organizations would be successful in procuring funding.

    - The features variables were APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, and ASK_AMT
    
    - The columns EIN and NAME were dropped from the data set rather than including them in the features as they are identifiers of the organizaitions that do not add insight into whether their request will be successful.  The SPECIAL_CONSIDERATIONS column was also dropped in an attempt to reduce noise in the data set after testing the initial model.

- Compiling, Training, and Evaluating the Model

    - I used 120 neurons for the two hidden layers of the model, as that was roughly 3 times the number of features.  I used two hidden layers and one output layer as one layer would likely not be sufficient in achieving an accurate model, and three layers proved no more effective than two layers while tweaking the model and took longer to fit.  I used the RELU activation function for the hidden layers because there were no negative values in the input data, and I used the Sigmoid activation function for the output layer to achieve a binary classification result.

    - The initial model had an accuracy of 72.76 %.  
    
    ![first model](Resources/first_accuracy.png)
    
    After attempting five times to optimize I was unable to achieve the desired 75 % accuracy.  The final version of the model tested achieved an accuracy of 73.07 %.

    ![final model](Resources/final_accuracy.png)

    - In attempting to optimize the model I first dropped an additional column from the features data set and grouped the requested donation amounts into bins to reduce noise in the data and used the same model structure, which yield a slightly higher accuracy.  For the next attempt at optimization I added more neurons to the two hidden layers.  The initial model used 80 neurons in the first layer-roughly twice the number of features variables-and 30 neurons in the second layer.  For this second attempt at optimization I increased to 90 and 45 neurons respectively.  

    For the third optimization attempt I used the same number of neurons in the first two hidden layers and added a third hidden layer.  This model's accuracy was comparable the previous one, so adding another layer did not improve performance and increased the time needed to fit the model.

    For the fourth attempt used the same number of neurons and hidden layers, but changed the activation function for the first two hidden layers from RELU to TANH.  This model had a lower accuracy than the previous, so for future attempts I returned to RELU activation functions.

    For the fifth and final attempt made here to optimize the model, I returned to RELU activation functions for hidden layers, reduced the number of layers back to two to limit the time necessary to fit the model, and increased the number of neurons from 90 and 45 for the hidden layers to 120 for both hidden layers-roughly three times the number of features.

    While this model still did not achieve the goal accuracy, it did perform the best of the models tested.  

## Summary

While the few attempts to optimize the initial Neural Network Deep Learning model did improve performance slightly, the model did not achieve the goal accuracy.  Continuing to eliminate noise in the data and alter the number of neurons, the number of layers, the activation functions of the model could further improve the performance of the model slightly, but at considerable cost of time and resources, and would likely not yield a significantly better performing model.

Because the desired result of the model is a simple binary classification, designing a RandomForestClassifier model might be more optimal.  It would take considerably less time to code and to train on the data, and could likely yield comparable accuracy results.  Also, the RandomForestClassifier as a model structure can be more easily described to and understood by non-data scientist stakeholders, making it more likely an organization would be willing to adopt its use.