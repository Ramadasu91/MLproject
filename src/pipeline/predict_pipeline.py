import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Temperature, Pressure, Humidity, WindDirection_Degrees, Speed, Month, Day, Hour, Minute, Second, risehour, riseminute, sethour, setminute):
        self.Temperature = Temperature
        self.Pressure = Pressure
        self.Humidity = Humidity
        self.WindDirection_Degrees = WindDirection_Degrees
        self.Speed = Speed
        self.Month = Month
        self.Day = Day
        self.Hour = Hour
        self.Minute = Minute
        self.Second = Second
        self.risehour = risehour
        self.riseminute = riseminute
        self.sethour = sethour
        self.setminute = setminute

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Temperature": [self.Temperature],
                "Pressure": [self.Pressure],
                "Humidity": [self.Humidity],
                "WindDirection(Degrees)": [self.WindDirection_Degrees],
                "Speed": [self.Speed],
                "Month": [self.Month],
                "Day": [self.Day],
                "Hour": [self.Hour],
                "Minute": [self.Minute],
                "Second": [self.Second],
                "risehour": [self.risehour],
                "riseminute": [self.riseminute],
                "sethour": [self.sethour],
                "setminute": [self.setminute]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
