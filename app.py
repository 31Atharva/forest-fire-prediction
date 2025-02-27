import pickle
import bz2
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from app_logger import log
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Import Classification and Regression model files
with bz2.BZ2File('Classification.pkl', 'rb') as C_pickle:
    model_C = pickle.load(C_pickle)

with bz2.BZ2File('Regression.pkl', 'rb') as R_pickle:
    model_R = pickle.load(R_pickle)


# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


# Route for Classification Model
@app.route('/predictC', methods=['POST'])
def predictC():
    try:
        # Reading inputs from the form
        Temperature = float(request.form['Temperature'])
        Wind_Speed = float(request.form['Wind_speed'])  # Ensuring consistency with form field name
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])

        features = [Temperature, Wind_Speed, FFMC, DMC, ISI]
        final_features = np.array([features])

        prediction = model_C.predict(final_features)[0]

        log.info('Prediction done for Classification model')

        text = 'Forest is Safe!' if prediction == 0 else 'Forest is in Danger!'
        return render_template('index.html', prediction_text1=f"{text} --- Chance of Fire is {prediction}")

    except Exception as e:
        log.error(f'Input error, check input: {str(e)}')
        return render_template('index.html', prediction_text1="Check the Input again!!!")


# Route for Regression Model
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        # Reading inputs from the form
        Temperature = float(request.form['Temperature'])
        Wind_Speed = float(request.form['Wind_speed'])  # Ensure consistency
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])

        features = [Temperature, Wind_Speed, FFMC, DMC, ISI]
        final_features = np.array([features])

        prediction = model_R.predict(final_features)[0]

        log.info('Prediction done for Regression model')

        hazard_status = "Warning!!! High hazard rating" if prediction > 15 else "Safe.. Low hazard rating"
        return render_template('index.html', prediction_text2=f"Fuel Moisture Code index is {prediction:.4f} ---- {hazard_status}")

    except Exception as e:
        log.error(f'Input error, check input: {str(e)}')
        return render_template('index.html', prediction_text2="Check the Input again!!!")


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=True, port=5000)
