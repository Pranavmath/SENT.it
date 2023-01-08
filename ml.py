# Imports
import os.path
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import BeitFeatureExtractor, BeitForImageClassification

path_beit = "OURSKINMODELS/beit/model"
path_google = "OURSKINMODELS/google/model"

# Loads Beit Model
classifier_beit = VisionClassifierInference(
    feature_extractor=BeitFeatureExtractor.from_pretrained(path_beit),
    model=BeitForImageClassification.from_pretrained(path_beit),
)

# Loads Google Model
classifier_google = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(path_google),
    model=ViTForImageClassification.from_pretrained(path_google),
)


# Given 2 tuples (Prediction, Probability) this function will get the Prediction that has higher probability
# If both tuples have the same prediction then it will just get the prediction
def predict_2_ensemble(top_a, top_b):
    if top_a[0] == top_b[0]:
        return top_a[0]
    else:
        return [max([top_a, top_b], key=lambda a: a[1])[0], max([top_a, top_b], key=lambda a: a[1])[1]]


# Uses the ensemble function to get the prediction and probability from applying the Beit and Google models on an image
def predict_ensemble(img_path):
    label_beit = classifier_beit.predict(img_path=img_path, return_str=False)
    label_google = classifier_google.predict(img_path=img_path, return_str=False)
    label_beit = {k: v for k, v in sorted(label_beit.items(), key=lambda item: item[1], reverse=True)}
    label_google = {k: v for k, v in sorted(label_google.items(), key=lambda item: item[1], reverse=True)}
    top_beit = list(label_beit.items())[0]
    top_google = list(label_google.items())[0]

    return predict_2_ensemble(top_beit, top_google)

# Given the PIL image save it and then apply the predict_ensemble
def predict(img):
    img.save("client_image.jpg")
    return predict_ensemble("client_image.jpg")
