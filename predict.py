import joblib
import numpy as np
import pandas as pd

encoder_BusinessTravel = joblib.load("model/encoder_BusinessTravel.joblib")
encoder_Department = joblib.load("model/encoder_Department.joblib")
encoder_Education = joblib.load("model/encoder_Education.joblib")
encoder_EducationField = joblib.load("model/encoder_EducationField.joblib")
encoder_EmployeeCount = joblib.load("model/encoder_EmployeeCount.joblib")
encoder_JobLevel = joblib.load("model/encoder_JobLevel.joblib")
encoder_JobRole = joblib.load("model/encoder_JobRole.joblib")
encoder_JobSatisfaction = joblib.load("model/encoder_JobSatisfaction.joblib")
encoder_MaritalStatus = joblib.load("model/encoder_MaritalStatus.joblib")
encoder_OverTime = joblib.load("model/encoder_OverTime.joblib")
encoder_PerformanceRating = joblib.load(
    "model/encoder_PerformanceRating.joblib")
encoder_StockOptionLevel = joblib.load("model/encoder_StockOptionLevel.joblib")
encoder_WorkLifeBalance = joblib.load("model/encoder_WorkLifeBalance.joblib")

scaler_Age = joblib.load("model/scaler_Age.joblib")
scaler_DailyRate = joblib.load("model/scaler_DailyRate.joblib")
scaler_DistanceFromHome = joblib.load("model/scaler_DistanceFromHome.joblib")
scaler_MonthlyIncome = joblib.load("model/scaler_MonthlyIncome.joblib")
scaler_MonthlyRate = joblib.load("model/scaler_MonthlyRate.joblib")
scaler_NumCompaniesWorked = joblib.load(
    "model/scaler_NumCompaniesWorked.joblib")
scaler_TotalWorkingYears = joblib.load("model/scaler_TotalWorkingYears.joblib")
scaler_YearsAtCompany = joblib.load("model/scaler_YearsAtCompany.joblib")
scaler_YearsInCurrentRole = joblib.load(
    "model/scaler_YearsInCurrentRole.joblib")
scaler_YearsWithCurrManager = joblib.load(
    "model/scaler_YearsWithCurrManager.joblib")
scaler_PercentSalaryHike = joblib.load(
    "model/scaler_PercentSalaryHike.joblib")


all_columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
               'Education', 'EducationField', 'EmployeeCount', 'JobLevel', 'JobRole',
               'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
               'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
               'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears',
               'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
               'YearsWithCurrManager']


def data_preprocessing(data):
    """PPreprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 

    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame(columns=all_columns)

    df["Age"] = scaler_Age.transform(np.asarray(
        data["Age"]).reshape(-1, 1)).reshape(-1,)
    df["DailyRate"] = scaler_DailyRate.transform(
        np.asarray(data["DailyRate"]).reshape(-1, 1)).reshape(-1,)
    df["DistanceFromHome"] = scaler_DistanceFromHome.transform(
        np.asarray(data["DistanceFromHome"]).reshape(-1, 1)).reshape(-1,)
    df["MonthlyIncome"] = scaler_MonthlyIncome.transform(
        np.asarray(data["MonthlyIncome"]).reshape(-1, 1)).reshape(-1,)
    df["NumCompaniesWorked"] = scaler_NumCompaniesWorked.transform(
        np.asarray(data["NumCompaniesWorked"]).reshape(-1, 1)).reshape(-1,)
    df["TotalWorkingYears"] = scaler_TotalWorkingYears.transform(
        np.asarray(data["TotalWorkingYears"]).reshape(-1, 1)).reshape(-1,)
    df["YearsAtCompany"] = scaler_YearsAtCompany.transform(
        np.asarray(data["YearsAtCompany"]).reshape(-1, 1)).reshape(-1,)
    df["YearsInCurrentRole"] = scaler_YearsInCurrentRole.transform(
        np.asarray(data["YearsInCurrentRole"]).reshape(-1, 1)).reshape(-1,)
    df["YearsWithCurrManager"] = scaler_YearsWithCurrManager.transform(
        np.asarray(data["YearsWithCurrManager"]).reshape(-1, 1)).reshape(-1,)
    df["MonthlyRate"] = scaler_MonthlyRate.transform(
        np.asarray(data["MonthlyRate"]).reshape(-1, 1)).reshape(-1,)
    df["PercentSalaryHike"] = scaler_PercentSalaryHike.transform(
        np.asarray(data["PercentSalaryHike"]).reshape(-1, 1)).reshape(-1,)

    df["BusinessTravel"] = encoder_BusinessTravel.transform(
        data["BusinessTravel"])
    df["Department"] = encoder_Department.transform(
        data["Department"])
    df["Education"] = encoder_Education.transform(
        data["Education"])
    df["EducationField"] = encoder_EducationField.transform(
        data["EducationField"])
    df["EmployeeCount"] = encoder_EmployeeCount.transform(
        data["EmployeeCount"])
    df["JobLevel"] = encoder_JobLevel.transform(
        data["JobLevel"])
    df["JobRole"] = encoder_JobRole.transform(
        data["JobRole"])
    df["JobSatisfaction"] = encoder_JobSatisfaction.transform(
        data["JobSatisfaction"])
    df["MaritalStatus"] = encoder_MaritalStatus.transform(
        data["MaritalStatus"])
    df["OverTime"] = encoder_OverTime.transform(
        data["OverTime"])
    df["PerformanceRating"] = encoder_PerformanceRating.transform(
        data["PerformanceRating"])
    df["StockOptionLevel"] = encoder_StockOptionLevel.transform(
        data["StockOptionLevel"])
    df["WorkLifeBalance"] = encoder_WorkLifeBalance.transform(
        data["WorkLifeBalance"])

    return df


model = joblib.load("model/xgb_model.joblib")


def prediction(data):
    """Making prediction

    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data

    Returns:
        numpy.ndarray: Prediction result (Churn or No)
    """
    result = model.predict(data)
    return result
