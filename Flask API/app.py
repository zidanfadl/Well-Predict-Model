# Import Lib
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import regex as re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

@app.route("/predict", methods={"POST"})
def predict():
    
    if request.method == "POST":

        ### Penting Start
        symptoms_data = request.json
        user_input = symptoms_data.get('symptoms', [])
        print(user_input)
        target_class = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis',
                        'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox',
                        'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)',
                        'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
                        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism',
                        'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis',
                        'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
                        'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']
        
        
        original_data = pd.read_csv('original_data.csv')

        def strip_to_basic_tokens(symptoms):
            symptoms = [symptom.strip().lower().replace(' ', '_').replace('_', ' ') for symptom in symptoms]
            return [re.sub(r'\s+', ' ', symptom) for symptom in symptoms]
        # Apply strip_to_basic_tokens function to user input
        user_input_stripped = strip_to_basic_tokens(user_input)

        # Initialize MultiLabelBinarizer with all symptoms
        mlb = MultiLabelBinarizer(classes=original_data.columns)

        # Fit and transform user input
        user_input_encoded = pd.DataFrame(mlb.fit_transform([user_input_stripped]), columns=mlb.classes_)

        # Concatenate user input with original data
        final_user_input = pd.concat([pd.DataFrame(columns=original_data.columns), user_input_encoded], axis=0)
        final_user_input = final_user_input.drop(['Disease'],axis = 1)
        user_tensor = tf.convert_to_tensor(final_user_input.values, dtype=tf.float32)
        


        predict_proba = model.predict(user_tensor)
        predicted_class_index = np.argmax(predict_proba)
        print(f'User tensor[0]: {user_tensor[0]}\nClass Index: {predicted_class_index}')
        prediction = target_class[predicted_class_index]
        print(prediction)
        ### Penting End

    return jsonify({"Prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
