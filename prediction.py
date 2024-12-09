import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

boston = pd.read_csv('boston.csv')
boston = boston.fillna(boston.mean())
boston.info()

x = boston.drop(columns='MEDV').values
y = boston['MEDV'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rand_forest = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, 20]
}

grid_search = GridSearchCV(estimator=rand_forest, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Cross Validation Score (R^2): {grid_search.best_score_}")

model = grid_search.best_estimator_

predictions = model.predict(x_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")


plt.figure(figsize=(10, 5))

plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.title('Random Forest Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

feature_names = boston.drop(columns='MEDV').columns
feature_importances = model.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(feature_names, feature_importances)
plt.title('Random Forest Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importances')
plt.show()
