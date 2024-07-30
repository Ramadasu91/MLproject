from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get the form values
            temperature = int(request.form.get('Temperature', 0))  # Default to 0 if None
            pressure = float(request.form.get('Pressure', 0.0))    # Default to 0.0 if None
            humidity = int(request.form.get('Humidity', 0))        # Default to 0 if None
            wind_direction = float(request.form.get('WindDirection(Degrees)', 0.0)) # Default to 0.0 if None
            speed = float(request.form.get('Speed', 0.0))          # Default to 0.0 if None
            month = int(request.form.get('Month', 1))              # Default to 1 if None
            day = int(request.form.get('Day', 1))                  # Default to 1 if None
            hour = int(request.form.get('Hour', 0))                # Default to 0 if None
            minute = int(request.form.get('Minute', 0))            # Default to 0 if None
            second = int(request.form.get('Second', 0))            # Default to 0 if None
            rise_hour = int(request.form.get('risehour', 0))       # Default to 0 if None
            rise_minute = int(request.form.get('riseminute', 0))   # Default to 0 if None
            set_hour = int(request.form.get('sethour', 0))         # Default to 0 if None
            set_minute = int(request.form.get('setminute', 0))     # Default to 0 if None

            # Create a CustomData instance with the form data
            data = CustomData(
                Temperature=temperature,
                Pressure=pressure,
                Humidity=humidity,
                WindDirection_Degrees=wind_direction,
                Speed=speed,
                Month=month,
                Day=day,
                Hour=hour,
                Minute=minute,
                Second=second,
                risehour=rise_hour,
                riseminute=rise_minute,
                sethour=set_hour,
                setminute=set_minute
            )
            
            # Convert to DataFrame and make predictions
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])

        except Exception as e:
            return str(e)

if __name__ == "__main__":
    app.run(debug=True)
