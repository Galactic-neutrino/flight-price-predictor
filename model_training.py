# Importing Libraries:
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from feature_engine.datetime import DatetimeFeatures


# Settings:
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output='pandas')


# Loading Data:
train_df = pd.read_csv(r'data\train.csv')
val_df = pd.read_csv(r'data\validation.csv')
test_df = pd.read_csv(r'data\test.csv')


# Splitting Data (d):
def split_data(d):
    x = d.drop(columns='Price')
    y = d.Price.copy()

    return x, y


x_train, y_train = split_data(train_df)
x_val, y_val = split_data(val_df)
x_test, y_test = split_data(test_df)


# Data Processing:
datetime_columns = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']
numeric_columns = ['Duration', 'Total_Stops']
categorical_columns = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']

doj_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("extractor", DatetimeFeatures(features_to_extract=['week', 'day_of_week', 'month', 'day_of_month'], format='mixed')),
    ("scaler", StandardScaler())
])
time_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("extractor", DatetimeFeatures(features_to_extract=['hour', 'minute'], format='mixed')),
    ("scaler", StandardScaler())
])
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessors = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, numeric_columns),
    ('categorical', categorical_transformer, categorical_columns),
    ('doj', doj_transformer, ['Date_of_Journey']),
    ('time', time_transformer, ['Dep_Time', 'Arrival_Time'])
])


# Curve Plotting Functions:
def plot_curves(sizes, mean_scores, std_scores, label, axis):
    axis.plot(sizes, mean_scores, marker='o', label=label)
    axis.fill_between(x=sizes, y1=mean_scores-std_scores, y2=mean_scores+std_scores, alpha=0.5)


def plot_learning_curves(algo_name, algo, figsize=(12, 4)):
    model = Pipeline(steps=[
        ("preprocessor", preprocessors),
        ("algorithm", algo)
    ])

    train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=x_data, y=y_data, cv=3, scoring='r2',
                                                            n_jobs=-1, random_state=42)

    mean_train_scores = np.mean(train_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)
    train_score = f"{mean_train_scores[-1]:.2f} +/- {std_train_scores[-1]:.2f}"

    mean_test_scores = np.mean(test_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)
    test_score = f"{mean_test_scores[-1]:.2f} +/- {std_test_scores[-1]:.2f}"

    figure, axis = plt.subplots(figsize=figsize)

    # training curve
    plot_curves(train_sizes, mean_train_scores, std_train_scores, f"Train ({train_score})", axis)

    # test curve
    plot_curves(train_sizes, mean_test_scores, std_test_scores, f"Test ({test_score})", axis)

    axis.set(xlabel="Training Set Sizes", ylabel="R-squared", title=algo_name)
    axis.legend(loc="lower right")

    plt.show()


# Model Selection:
algorithms = {
    "Linear Regression": LinearRegression(),
    "Support Vector Machine": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=10),
    "XG Boost": XGBRegressor(n_estimators=10)
}

data = pd.concat([train_df, val_df], axis=0)
x_data, y_data = split_data(data)

for algorithm_name, algorithm in algorithms.items():
    plot_learning_curves(algorithm_name, algorithm)


# Model Training:
model = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('algorithm', RandomForestRegressor(n_estimators=10))
])
model.fit(x_data, y_data)


# Model Evaluation:
def evaluate_model(x, y):
    y_pred = model.predict(x)
    return r2_score(y, y_pred)


print(f"The R2 score on training data is: {evaluate_model(x_data, y_data)}")
print(f"The R2 score on test data is: {evaluate_model(x_test, y_test)}")


# Model Persistence:
saved_model = model
joblib.dump(saved_model, 'model.joblib')


# Testing:
testing_model = joblib.load(saved_model)
y_predict = saved_model.predict(x_test)
score = r2_score(y_test, y_predict)
print(score)
