import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from scipy import stats
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
import re

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = [
                "Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", 
                "Speed", "Month", "Day", "Hour", "Minute", "Second", 
                "risehour", "riseminute", "sethour", "setminute"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", RobustScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def apply_transformations(self, df):
        '''
        Apply specified transformations to the dataframe
        '''
        try:
            # Apply log transformation to Temperature and Speed
            df['Temperature'] = np.log(df['Temperature'] + 1)
            df['Speed'] = np.log(df['Speed'] + 1)

            # Apply Box-Cox transformation to Pressure and Humidity
            df['Pressure'], _ = stats.boxcox(df['Pressure'] + 1)
            df['Humidity'], _ = stats.boxcox(df['Humidity'] + 1)

            # Apply MinMaxScaler to WindDirection
            df['WindDirection(Degrees)'] = MinMaxScaler().fit_transform(
                np.array(df['WindDirection(Degrees)']).reshape(-1, 1)
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Split the 'Data' column into 'Date' and 'Time'
            train_df['Date'] = train_df['Data'].apply(lambda x: x.split()[0])
            train_df['Time'] = train_df['Data'].apply(lambda x: x.split()[1])
            test_df['Date'] = test_df['Data'].apply(lambda x: x.split()[0])
            test_df['Time'] = test_df['Data'].apply(lambda x: x.split()[1])

            # Extract month, day from 'Date' and hour, minute, second from 'Time'
            train_df['Month'] = pd.to_datetime(train_df['Date']).dt.month
            train_df['Day'] = pd.to_datetime(train_df['Date']).dt.day
            train_df['Hour'] = pd.to_datetime(train_df['Time']).dt.hour
            train_df['Minute'] = pd.to_datetime(train_df['Time']).dt.minute
            train_df['Second'] = pd.to_datetime(train_df['Time']).dt.second

            test_df['Month'] = pd.to_datetime(test_df['Date']).dt.month
            test_df['Day'] = pd.to_datetime(test_df['Date']).dt.day
            test_df['Hour'] = pd.to_datetime(test_df['Time']).dt.hour
            test_df['Minute'] = pd.to_datetime(test_df['Time']).dt.minute
            test_df['Second'] = pd.to_datetime(test_df['Time']).dt.second

            # Extract risehour, riseminute from 'TimeSunRise' and sethour, setminute from 'TimeSunSet'
            train_df['risehour'] = train_df['TimeSunRise'].apply(lambda x: re.search(r'^\d+', x).group(0)).astype(int)
            train_df['riseminute'] = train_df['TimeSunRise'].apply(lambda x: re.search(r'(?<=\:)\d+(?=\:)', x).group(0)).astype(int)
            train_df['sethour'] = train_df['TimeSunSet'].apply(lambda x: re.search(r'^\d+', x).group(0)).astype(int)
            train_df['setminute'] = train_df['TimeSunSet'].apply(lambda x: re.search(r'(?<=\:)\d+(?=\:)', x).group(0)).astype(int)

            test_df['risehour'] = test_df['TimeSunRise'].apply(lambda x: re.search(r'^\d+', x).group(0)).astype(int)
            test_df['riseminute'] = test_df['TimeSunRise'].apply(lambda x: re.search(r'(?<=\:)\d+(?=\:)', x).group(0)).astype(int)
            test_df['sethour'] = test_df['TimeSunSet'].apply(lambda x: re.search(r'^\d+', x).group(0)).astype(int)
            test_df['setminute'] = test_df['TimeSunSet'].apply(lambda x: re.search(r'(?<=\:)\d+(?=\:)', x).group(0)).astype(int)

            # Drop the original 'Data', 'Date', 'Time', 'TimeSunRise', 'TimeSunSet', and 'UNIXTime' columns
            train_df = train_df.drop(columns=['Data', 'Date', 'Time', 'TimeSunRise', 'TimeSunSet', 'UNIXTime'])
            test_df = test_df.drop(columns=['Data', 'Date', 'Time', 'TimeSunRise', 'TimeSunSet', 'UNIXTime'])

            # Convert the target column values by multiplying with 100
            train_df['Radiation'] = train_df['Radiation'].apply(lambda x: int(x * 100))
            test_df['Radiation'] = test_df['Radiation'].apply(lambda x: int(x * 100))

            # Apply specified transformations
            train_df = self.apply_transformations(train_df)
            test_df = self.apply_transformations(test_df)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Radiation"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit on training data and transform both training and testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
