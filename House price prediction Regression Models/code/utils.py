######################################################
# This script summarizes all the functions we have developed
# during the project, and automatically loads
# all the transformed dataset for testing
######################################################


print("Loading functions and raw data")
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from numpy import mean, std, absolute
from scipy import stats

from sklearn.metrics import mean_absolute_error

import random

# Load data
data_dirpath = 'C:\\Users\\yixin\\Desktop\\Workspace\\Personal GitHub Repos\\House-Prices\\Projects\\House price prediction\\data'

train_csv = pd.read_csv(os.path.join(data_dirpath, "train.csv"))
test_csv = pd.read_csv(os.path.join(data_dirpath, "test.csv"))

# Maps to store original information of the dataframe (e.g.: datatype, column name)

# We need to store these information because once the dataframe is labelled, the column namd and the datatype will be lost

train_csv = train_csv.drop(columns=["Id"])

col_name_map = dict()

for index, col_name in enumerate(train_csv.columns):
    col_name_map.setdefault(index, col_name)


def change_datatype(df, col_names, new_type):
    for col_name in col_names:
        df[col_name] = df[col_name].astype(new_type)

    return df


def label_encoder(df, start_index = 0):
    label_encoder = LabelEncoder()

    for col_name in df.columns:
        if df[col_name].dtype == "O":
            df[col_name] = label_encoder.fit_transform(df[col_name]) + start_index

    return df


def iterative_impute(df, max_iter=100, random_state=0, verbose=True):
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state, verbose=verbose)

    df = imp.fit_transform(df)
    df = pd.DataFrame(df)

    return df

train_csv = change_datatype(train_csv, ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd","GarageYrBlt"], str)

datatype_map = dict()

for index, col_name in enumerate(train_csv.columns):
    datatype_map.setdefault(index,str(train_csv[col_name].dtype))


def display_boxplot(df):
    for col_nr in df.columns:

        if datatype_map[col_nr] != "object":
            sns.barplot(x=col_nr, data=df)
        plt.title(f"boxplot of {col_name_map[col_nr]}")
        plt.show()


def display_distplot(df):
    for col_nr in df.columns:

        if datatype_map[col_nr] != "object":
            sns.distplot(df[col_nr])
        plt.title(f"Distplot of {col_name_map[col_nr]}")
        plt.show()


def display_scatter(df):
    for col_nr in df.columns[:-1]:

        if datatype_map[col_nr] != "object":
            plt.scatter(df[col_nr], df.iloc[:, -1:])

        plt.title(f"Scatterplot of of {col_name_map[col_nr]}")
        plt.xlabel(col_name_map[col_nr])
        plt.ylabel("House price")
        plt.show()


def data_transform(df, mode="min-max"):
    assert any([mode == "min-max", mode == "abs-max", mode == "z-score", mode == "log", mode == "yeo-johnson", mode == "sqrt"]), "Invalid mode name. \
        Please enter 'min-max', 'abs-max', 'z-score', 'log','sqrt' or 'yeo-johnson'."

    df_transformed = df.copy()

    for column in df_transformed.columns[:-1]:

        # min-max scale
        if mode == "min-max":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].min()) / (
                        df_transformed[column].max() - df_transformed[column].min())

        # Absolute max value
        elif mode == "abs-max":
            df_transformed[column] = df_transformed[column] / df_transformed[column].abs().max()

        # z-score
        elif mode == "z-score":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].mean()) / df_transformed[
                column].std()

        # log normalization
        elif mode == "log":
            df_transformed[column] = np.log(df_transformed[column] + 1)

        elif mode == "yeo-johnson":
            try:
                transformed_data, _ = stats.yeojohnson(np.array(df_transformed[column]))
            except Exception as e:
                print(f"Yeo-Johnson transformation failed for column {column}: {e}")
            df_transformed[column] = transformed_data

        # Sqrt
        elif mode == "sqrt":
            df_transformed[column] = np.sqrt(df_transformed[column])


    return df_transformed

def data_transform_test(df, mode="min-max"):
    assert any([mode == "min-max", mode == "abs-max", mode == "z-score", mode == "log", mode == "yeo-johnson", mode == "sqrt"]), "Invalid mode name. \
        Please enter 'min-max', 'abs-max', 'z-score', 'log', 'sqrt' or 'yeo-johnson'."

    df_transformed = df.copy()

    for column in df_transformed.columns:

        # min-max scale
        if mode == "min-max":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].min()) / (
                        df_transformed[column].max() - df_transformed[column].min())

        # Absolute max value
        elif mode == "abs-max":
            df_transformed[column] = df_transformed[column] / df_transformed[column].abs().max()

        # z-score
        elif mode == "z-score":
            df_transformed[column] = (df_transformed[column] - df_transformed[column].mean()) / df_transformed[
                column].std()

        # log normalization
        elif mode == "log":
            df_transformed[column] = np.log(df_transformed[column] + 1)

        # Yeo-johnson
        elif mode == "yeo-johnson":
            try:
                transformed_data, _ = stats.yeojohnson(np.array(df_transformed[column]))
            except Exception as e:
                print(f"Yeo-Johnson transformation failed for column {column}: {e}")
            df_transformed[column] = transformed_data

        # Sqrt
        elif mode == "sqrt":
            df_transformed[column] = np.sqrt(df_transformed[column])

    return df_transformed

def remove_outliers(df, iqr_range=5):
    df_new = df.copy()

    outlier_index = set()

    for col_name in df_new.columns[:-1]:
        if datatype_map[col_name] != "object":

            q1 = df_new[col_name].quantile(0.25)
            q3 = df_new[col_name].quantile(0.75)

            iqr = q3 - q1

            outliers = df_new[(df_new[col_name] < (q1 - iqr_range * iqr)) | (df_new[col_name] > (q3 + iqr_range * iqr))]

            for row_index in outliers.index:
                outlier_index.add(row_index)

    df_new = df_new.drop(index=list(outlier_index))

    return df_new


def remove_outliers_capping(df, lower_bond=0.01, upper_bond=0.99):
    df_new = df.copy()

    outlier_index = set()

    for col_name in df_new.columns[:-1]:

        if datatype_map[col_name] != "object":
            lower_percentile = df_new[col_name].quantile(lower_bond)
            upper_percentile = df_new[col_name].quantile(upper_bond)

            outliers = df_new[(df_new[col_name] < lower_percentile) | (df_new[col_name] > upper_percentile)]

            for row_index in outliers.index:
                outlier_index.add(row_index)

    df_new = df_new.drop(index=list(outlier_index))

    return df_new


def test_model(df, random_state = 50):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    alphas = np.logspace(-4, 0, 50)

    result_mean = []
    result_std = []
    result_alpha = []

    for alpha in alphas:
        model = Lasso(alpha=alpha)

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state = random_state)
        # evaluate model
        scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

        print(f'Mean MAE: {mean(scores):.3f} STD: {std(scores):.3f} alpha={alpha}')

        result_mean.append(mean(scores))
        result_std.append(std(scores))
        result_alpha.append(alpha)

    print(" ")

    print(f"Maximum MAE: {max(result_mean)}, alpha={result_alpha[result_mean.index(max(result_mean))]}")
    print(f"Minimum MAE: {min(result_mean)}, alpha={result_alpha[result_mean.index(min(result_mean))]}")


def test_model_rmse(df, random_state = 50, result_to_txt = False, df_name = None):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    # alphas = np.arange(1, 100, 1).tolist()
    alphas = np.concatenate((np.logspace(-4,0,20), np.arange(1, 100, 1).tolist()))

    result_mean = []
    result_std = []
    result_alpha = []

    # Create a scorer for RMSE
    rmse_scorer = make_scorer(mean_squared_error, squared=False)  # squared=False gives RMSE

    for alpha in alphas:
        model = Lasso(alpha=alpha)

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state= random_state)
        # Evaluate model using RMSE
        scores = cross_val_score(model, X, Y, scoring=rmse_scorer, cv=cv, n_jobs=-1)

        print(f'Mean RMSE: {np.mean(scores):.3f} STD: {np.std(scores):.3f} alpha={alpha}')

        result_mean.append(np.mean(scores))
        result_std.append(np.std(scores))
        result_alpha.append(alpha)

    print(" ")

    print(f"Maximum RMSE: {max(result_mean):.3f}, alpha={result_alpha[result_mean.index(max(result_mean))]}")
    print(f"Minimum RMSE: {min(result_mean):.3f}, alpha={result_alpha[result_mean.index(min(result_mean))]}")

    if result_to_txt:
        best_record_sentence = f"{df_name}: {max(result_mean):.3f}, alpha={result_alpha[result_mean.index(min(result_mean))]} \n"
        with open("result.txt", "a") as outfile:
            outfile.write(best_record_sentence)


def test_elasticnet_gscv(df, df_name=None, result_to_txt=False):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    alphas = np.concatenate((np.logspace(-4, 0, 10), np.arange(1, 100, 10).tolist()))
    max_iter = list(range(1000, 10001, 1000))
    l1_ratio = np.arange(0.0, 1.0, 0.05).tolist()
    tol = [0.0001, 0.01, 0.1, 0.5]

    param_grid = {'alpha': alphas,
                  'max_iter': max_iter,
                  'l1_ratio': l1_ratio,
                  'tol': tol,
                  'random_state': [50]}

    elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, scoring='neg_root_mean_squared_error',
                                   cv=5, n_jobs=-1)

    elasticnet_gscv.fit(X, Y)

    print(f"Best parameter grids: {elasticnet_gscv.best_params_}")
    print(f"Score: {elasticnet_gscv.best_score_}")

    if result_to_txt:
        best_record_sentence = f"{df_name}: {elasticnet_gscv.best_score_}, {elasticnet_gscv.best_params_} \n"

        with open("result.txt", "a") as outfile:
            outfile.write(best_record_sentence)
def check_residuals(data, alpha, model = "lasso", l1_ratio = None, max_iter = None, tol=None, savefig=""):

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

    if model.lower() == 'lasso':
        model_residual = Lasso(alpha=alpha)

    elif model.lower() == 'elastic-net':
        assert all([l1_ratio is not None, max_iter is not None, tol is not None]), \
            "You are using Elastic-net model. For this model, please pass in the l1-ratio, max iter and tol as well."

        model_residual = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter = max_iter,
            tol = tol)

    else:
        raise ValueError(f"Unknown model: {model}. Please choose 'lasso' or 'elastic-net'.")

    model_residual.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model_residual.predict(X_test)

    # Calculate the residuals
    residuals = Y_test - Y_pred

    print(residuals.shape, Y_test.shape, Y_pred.shape)

    plt.figure(figsize=(8, 6))
    plt.scatter(Y_pred, residuals, color='blue', edgecolor='k')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model} Regression (alpha={alpha})')

    if savefig:
        plt.savefig(savefig)

    plt.show()

def display_corr_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(30,30))
    sns.heatmap(corr)
    plt.show()


def display_pairplot(df):
    i = 0
    while i < 78:
        first_columns = train_csv.iloc[:, i:i + 6]
        last_column = train_csv.iloc[:, -1]

        sub_df = pd.concat([first_columns, last_column], axis=1)
        sns.pairplot(sub_df)
        plt.show()
    i += 6
### Preprocess and transform data

## Original training data

print("Processing training data")
# Change datatype, impute missing values and encode categorical data
train_csv = change_datatype(train_csv, ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd","GarageYrBlt"], str)
train_csv = label_encoder(train_csv)
train_csv = iterative_impute(train_csv)

# Transform data
train_minmax_transformed = data_transform(train_csv, "min-max")
train_absmax_transformed = data_transform(train_csv, "abs-max")
train_zscore_transformed = data_transform(train_csv, "z-score")
train_log_transformed = data_transform(train_csv, "log")
train_yj_transformed = data_transform(train_csv, "yeo-johnson")

train_nocap = remove_outliers(train_csv)
train_log_transformed_nocap = remove_outliers(train_log_transformed)
train_minmax_transformed_nocap = remove_outliers(train_minmax_transformed)
train_absmax_transformed_nocap = remove_outliers(train_absmax_transformed)
train_zscore_transformed_nocap = remove_outliers(train_zscore_transformed)
train_yj_transformed_nocap = remove_outliers(train_yj_transformed)

train_capped = remove_outliers_capping(train_csv)
train_log_transformed_capped = remove_outliers_capping(train_log_transformed)
train_minmax_transformed_capped = remove_outliers_capping(train_minmax_transformed)
train_absmax_transformed_capped = remove_outliers_capping(train_absmax_transformed)
train_zscore_transformed_capped = remove_outliers_capping(train_zscore_transformed)
train_yj_transformed_capped = remove_outliers_capping(train_yj_transformed)

# Preprocess test data
print("Processing test data")
test_csv_ids = test_csv["Id"]
test_csv = change_datatype(test_csv, ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd","GarageYrBlt"], str)
test_csv = test_csv.drop(columns=["Id"])
test_csv = label_encoder(test_csv)
test_csv = iterative_impute(test_csv)

test_minmax_transformed = data_transform_test(test_csv, "min-max")
test_absmax_transformed = data_transform_test(test_csv, "abs-max")
test_zscore_transformed = data_transform_test(test_csv, "z-score")
test_log_transformed = data_transform_test(test_csv, "log")
test_yj_transformed = data_transform_test(test_csv, "yeo-johnson")

print("Data Loading and preprocessing complete!")