from flask import Flask, request, render_template
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log form data
        logging.debug(f"Form data received: {request.form}")

        # Extract form data
        step = int(request.form['step'])
        customer = int(request.form['customer'])
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        merchant = int(request.form['merchant'])
        category = int(request.form['category'])
        amount = float(request.form['amount'])
        
        # Prepare input for the model
        input_data = [[step, customer, age, gender, merchant, category, amount]]
        logging.debug(f"Input data for prediction: {input_data}")
        
        # Make prediction
        prediction = model.predict(input_data)
        result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
        logging.debug(f"Prediction result: {result}")
        
        return render_template('index.html', prediction_text=f"Prediction: {result}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
