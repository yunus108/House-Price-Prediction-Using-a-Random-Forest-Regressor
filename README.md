# House Price Prediction Using a Random Forest Regressor

In this project I created a random forest regressor model to predict the house prices in the boston housing prices dataset.
The dataset consists of 506 entries and 14 columns which means we have 13 features and a label column.
I chose to use a random forest regressor because the relationship between each feature and our target variable could not be highly linear so i wanted to use a model that would be a bit better at understanding complex relationships.

I applied hyperparameter tuning using grid search and the result for the best hyperparameters were 100 for n_estimators and 15 for max_depth. The other parameters were left at their default values.
Cross validation score of the model is 0.82 meaning the model is able to explain the variance fairly well and successfully capturing the patterns.
Mean Squared Error on the test set is 7.93 which indicates a good performance result for the regression task at hand.

I plotted the actual and the predicted values to see how the model performs in a graphical sense. 
I also plotted the feature importances to understand how the model interpretes the data and which features have more impact on the predictions.

## Requirements
pandas
scikit-learn
matplotlib

## Acknowledgements
Dataset i used : <https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data>
