# Explainer

## Features(X) vs Label(y)
- features are the independent variables, and what we know about the data.
- label is dependent variable, and what we want to predict.
- features are what we extract from data, that will help us in predicting the label.
- features to label relationship is many to one, meanng many features combine to predict the label, but one feature can also predict the label sometimes when complexity is not high.

## Train / validation / test split (why we split three ways and never let the model see test data during training)

- we use train, validation and test split to evaluate the model's ability to learn the data and generalize to unseen data.
- train set is used to train the model.
- validation set is used to tune the model hyperparameters.
- test set is used to evaluate the model's performance on unseen data.
- generally, we use 60% of the data for train, 20% for validation and 20% for test. This is the general convention, but it can be changed according to the data size and complexity.
- to avoid overfitting and underfitting

##  Overfitting vs. underfitting
- overfitting is when our model has learnt the training data (100% train accuracy), but fails badly on unseen data (validation set).
- overfitting occurs when model is too complex and has too many parameters, or training data is too small.

- underfitting is when our model fails to learn the training data (low train accuracy), and also fails on unseen data (bad on both train and validation set).
- underfitting occurs when model is too simple and has too few parameters, or training data is too small.

- our goal is to be between these two extremes, we want the model to learn the training data well, but not too well that it fails on unseen data.
- a good model will have high train accuracy and high validation accuracy.
- so keeping a good training set size and good feature engineering can help us achieve this.

## Bias–variance trade-off
- bias is the error that occurs when a model is too simple to capture the underlying pattern in the data.
- variance is the error that occurs when a model is too complex and captures the noise in the data.
- when we try to increase the model's accuracy (by increasing the model's complexity), we increase its ability to capture the underlying pattern, but at the same time, we increase its ability to capture the noise in the data.
- how we balance this trade-off is by using cross validation and regularization.
- Bias (Underfitting): The model makes rigid, wrong assumptions (like assuming a wave is a flat line).
- Variance (Overfitting): The model is highly unstable and changes completely based on minor noise variations.