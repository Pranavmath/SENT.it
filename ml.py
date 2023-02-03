import numpy as np
import tensorflow as tf
from tensorflow import keras

saved_model = keras.models.load_model("model")

from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import BeitFeatureExtractor, BeitForImageClassification

path_beit = "OURSKINMODELS/beit/model"
path_google = "OURSKINMODELS/google/model"

# Loads Models
classifier_beit = VisionClassifierInference(
    feature_extractor=BeitFeatureExtractor.from_pretrained(path_beit),
    model=BeitForImageClassification.from_pretrained(path_beit),
)

classifier_google = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(path_google),
    model=ViTForImageClassification.from_pretrained(path_google),
)


# Givens 2 tuples: (Prediction, Probability) it gets the Tuple with higher probability
def predict_2_ensemble(top_a, top_b):
    if top_a[0] == top_b[0]:
        return top_a[0]
    else:
        return [max([top_a, top_b], key=lambda a: a[1])[0], max([top_a, top_b], key=lambda a: a[1])[1]]


# Applies the ensemble to the 2 models
def ensemble_2(img_path):
    label_beit = classifier_beit.predict(img_path=img_path, return_str=False)
    label_google = classifier_google.predict(img_path=img_path, return_str=False)
    label_beit = {k: v for k, v in sorted(label_beit.items(), key=lambda item: item[1], reverse=True)}
    label_google = {k: v for k, v in sorted(label_google.items(), key=lambda item: item[1], reverse=True)}
    top_beit = list(label_beit.items())[0]
    top_google = list(label_google.items())[0]

    return predict_2_ensemble(top_beit, top_google)


# Given PIL Image
def predict(img):
    img.save("client_image.jpg")
    return ensemble_2("client_image.jpg")

list_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
                 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
                 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
                 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
                 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
                 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
                 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
                 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
                 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
                 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
                 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
                 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
                 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
                 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
                 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
                 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
                 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
                 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
                 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

onehot_encoder_inputs = {l: i for i, l in enumerate(list_symptoms)}
onehot_encoder_labels = {'Fungal infection': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Allergy': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bronchial Asthma': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Jaundice': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Malaria': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Chicken pox': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Dengue': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Typhoid': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'Tuberculosis': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Common Cold': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Pneumonia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'Varicose veins': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], '(vertigo) Paroymsal  Positional Vertigo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Acne': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Psoriasis': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Impetigo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


def onehot_encode_inputs(inputs):
    inital = [0] * 132

    for input in inputs:
        inital[onehot_encoder_inputs[input]] = 1

    return inital


# Given numpy array with shape (132, ) return the prediction
def predict_array(nparr):
    if nparr.dtype != np.float32:
        raise Exception("Your input numpy array must have a dtype of float32")

    tensor = tf.convert_to_tensor([nparr])
    predict = saved_model.predict(tensor)

    predict_list = list(predict)[0]

    sorted_predict = sorted({i: predict_list[i] for i in range(len(predict_list))}.items(), key=lambda i: i[1],
                            reverse=True)

    index_prediction = sorted_predict[0][0]

    prediction = list(onehot_encoder_labels.keys())[index_prediction]

    return prediction
