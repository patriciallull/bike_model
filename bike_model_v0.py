import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import joblib
import pathlib
from pathlib import Path
import os.path
from os import path
from sklearn.model_selection import GridSearchCV


# Load the data
print("Please make sure the data file 'hour.csv' is your current directory.")
hour = pd.read_csv("./hour.csv", index_col="instant", parse_dates=True)


# Model
def train_and_persist(hour):
    print("Pre-processing the data...")
    # creating duplicate columns for feature engineering
    hour["hr2"] = hour["hr"]
    hour["season2"] = hour["season"]
    hour["temp2"] = hour["temp"]
    hour["hum2"] = hour["hum"]
    hour["weekday2"] = hour["weekday"]
    # Change dteday to date time
    hour["dteday"] = pd.to_datetime(hour["dteday"])
    # Convert the data type to eithwe category or to float
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in int_hour:
        hour[col] = hour[col].astype("category")
    # change for skeweness
    hour["windspeed"] = np.log1p(hour.windspeed)
    hour["cnt"] = np.sqrt(hour.cnt)
    print("Doing some feature engineering...")
    ##### FEATURE ENGINEERING
    # Rented during office hours
    hour["IsOfficeHour"] = np.where(
        (hour["hr2"] >= 9) & (hour["hr2"] < 17) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsOfficeHour"] = hour["IsOfficeHour"].astype("category")
    # Rented during daytime
    hour["IsDaytime"] = np.where((hour["hr2"] >= 6) & (hour["hr2"] < 22), 1, 0)
    hour["IsDaytime"] = hour["IsDaytime"].astype("category")
    # Rented during morning rush hour
    hour["IsRushHourMorning"] = np.where(
        (hour["hr2"] >= 6) & (hour["hr2"] < 10) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourMorning"] = hour["IsRushHourMorning"].astype("category")
    # Rented during evening rush hour
    hour["IsRushHourEvening"] = np.where(
        (hour["hr2"] >= 15) & (hour["hr2"] < 19) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourEvening"] = hour["IsRushHourEvening"].astype("category")
    # Rented during most busy season
    hour["IsHighSeason"] = np.where((hour["season2"] == 3), 1, 0)
    hour["IsHighSeason"] = hour["IsHighSeason"].astype("category")
    # binning temp, atemp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    hour["temp_binned"] = pd.cut(hour["temp2"], bins).astype("category")
    hour["hum_binned"] = pd.cut(hour["hum2"], bins).astype("category")
    # dropping duplicated rows used for feature engineering
    hour = hour.drop(columns=["hr2", "season2", "temp2", "hum2", "weekday2"])
    ## get dummies
    hour = pd.get_dummies(hour)
    # return hour
    print("Splitting into train and test...")
    # hour = preprocess_data(hour)
    hour_train = hour.iloc[0:15211]
    hour_test = hour.iloc[15212:17379]
    ## round
    round(hour_train.describe(), 3)
    ######### STARTING
    train = hour_train.drop(columns=["dteday", "casual", "atemp", "registered"])
    test = hour_test.drop(columns=["dteday", "casual", "registered", "atemp"])
    # return hour_train, hour_test, train, test
    print("Training and fitting the model...")
    # seperate the independent and target variable on trainig data
    # hour_train, hour_test, train, test = split_train_test(hour)
    train_X = train.drop(columns=["cnt"], axis=1)
    train_y = train["cnt"]
    # seperate the independent and target variable on testing data
    test_X = test.drop(columns=["cnt"], axis=1)
    test_y = test["cnt"]
    # return train_X, train_y, test_X, test_y
    # RANDOM FOREST
    # Grid search

    # gsc = GridSearchCV(
    #    estimator=RandomForestRegressor(),
    #   param_grid={ 'max_depth': [10, 40, ],
    #               'min_samples_leaf': [1, 2],
    #              'min_samples_split': [2, 5],
    #             'n_estimators': [200, 400]},
    # cv=5,
    # scoring="r2",
    # verbose=10,
    # n_jobs=4,
    # )

    # grid_result = gsc.fit(train_X, train_y)

    # gsc.best_params_

    rf = RandomForestRegressor(
        max_depth=40,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42,
    )
    rf.fit(train_X, train_y)
    result = rf.predict(test_X)

    print("R-squared for Train: %.2f" % rf.score(train_X, train_y))
    print("R-squared for Test: %.2f" % rf.score(test_X, test_y))

    RMSE = np.sqrt(np.mean((test_y ** 2 - result ** 2) ** 2))
    MSE = RMSE ** 2

    print("MSE ={}".format(MSE))
    print("RMSE = {}".format(RMSE))
    # return rf
    # rf = model(hour)
    rf.fit(train_X, train_y)
    result_rf = rf.predict(test_X)
    print("R-squared for Train: %.2f" % rf.score(train_X, train_y))
    print("R-squared for Test: %.2f" % rf.score(test_X, test_y))
    print("To save model on disk, un-comment the next 4 lines in the code! ")
    # path_name = str(Path.home())
    # name = 'model_v0.pkl'
    # complete_path_file = path_name + '/' + name
    # joblib.dump(rf, complete_path_file)
    return rf, result_rf
    # return test_X, test_y, rf, result_rf


# Using the model
train_and_persist(hour)
