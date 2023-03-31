from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

labels = ['Acne and Rosacea Photos', 'Actinic cheilitis', 'Normal Skin', 'acanthosis nigricans', 'actinic keratosis',
          'alopecia', 'angiokeratoma', 'atopic dermatitis', 'atypical melanocytic proliferation',
          'basal cell carcinoma', 'biting insects', 'bowens disease', 'bullous disease', 'candida', 'candidiasis',
          'chondrodermatitis nodularis', 'ctcl', 'cutaneceous larva migrans', 'dermatofibroma',
          'distal subungual onychomycosis', 'drug eruptions', 'eczema', 'epidermal cyst', 'fixed drug eruption',
          'folliculitis', 'granuloma annulare', 'hemangioma', 'herpes', 'impetigo', 'intertrigo', 'keloids',
          'keratoacanthoma', 'lentigo', 'lichen planus', 'lichenoid keratosis', 'lupus', 'melanocytic nevi', 'melanoma',
          'molluscum contagiosum', 'necrobiosis lipoidica', 'neurofibromatosis', 'nevus',
          'other connective tissue diseases', 'other lichen related diseases', 'other light diseases',
          'other nail related diseases', 'other psoraisis related diseases', 'perleche', 'pigmented benign keratosis',
          'pityriasis and related diseases', 'poison ivy related diseases', 'porokeratosis', 'psoriasis', 'pubic lice',
          'pyogenic granuloma', 'scabies', 'scar', 'sebaceous hyperplasia', 'seborrheic dermatitis',
          'seborrheic keratosis', 'skin tag polyps', 'solar lentigo', 'squamous cell carcinoma', 'stucco keretoses',
          'sunburn', 'syringoma', 'telangiectasias', 'tick bite', 'tinea', 'tuberous sclerosis', 'urticaria',
          'varicella', 'vascular lesion', 'vasculitis', 'venous diseases', 'viral exanthems', 'warts', 'xanthomas']

vit_model = ViTForImageClassification.from_pretrained(
    "skinmodels/vit-base-SKINMODEL",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
)

deit_model = ViTForImageClassification.from_pretrained(
    "skinmodels/deit-base-SKINMODEL",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
)

vit_feature_extractor = ViTFeatureExtractor.from_pretrained("skinmodels/vit-base-SKINMODEL")

deit_feature_extractor = ViTFeatureExtractor.from_pretrained("skinmodels/deit-base-SKINMODEL")


# Keeps a from 0 to 1
def norm(a):
    return min(max(a, 0), 1)


def run_vit(PIL_Image):
    inputs = vit_feature_extractor(images=PIL_Image, return_tensors="pt")
    outputs = vit_model(**inputs)
    logits = outputs.logits

    top_5_class_idx = np.array(logits.argsort())[0][::-1][:5]

    top_5_class = [(vit_model.config.id2label[str(class_idx)],
                    norm(logits[0][class_idx].item() / 10)) for class_idx in top_5_class_idx]

    return top_5_class


def run_deit(PIL_Image):
    inputs = deit_feature_extractor(images=PIL_Image, return_tensors="pt")
    outputs = deit_model(**inputs)
    logits = outputs.logits

    top_5_class_idx = np.array(logits.argsort())[0][::-1][:5]

    top_5_class = [(vit_model.config.id2label[str(class_idx)],
                    norm(logits[0][class_idx].item() / 10)) for class_idx in top_5_class_idx]

    return top_5_class


def ensemble_2(t1, t2):
    if t1[0] == t2[0]:
        return (t1[0], (t1[1] + t2[1]) / 2)
    else:
        if t1[1] >= t2[1]:
            return (t1[0], t1[1])
        if t1[1] < t2[1]:
            return (t2[0], t2[1])


def predict(PIL_Image):
    deit_predicition = run_deit(PIL_Image)
    vit_prediction = run_vit(PIL_Image)

    ensemble_prediction = []

    for idx in range(len(vit_prediction)):
        ensemble_prediction.append(ensemble_2(vit_prediction[idx], deit_predicition[idx]))

    return ensemble_prediction


# --------------------------------
# Symptoms model below:

import tensorflow_hub as hub
from sklearn import metrics
import gzip
import json
import scipy
import numpy as np
import warnings
from time import time
from numpy.linalg import norm
from numba import njit, prange, jit
import pandas as pd
import pickle
from transformers import pipeline
import math

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


@njit
def cosine_similarity(np_array_a, np_array_b):
    return np.dot(np_array_a, np_array_b) / (norm(np_array_a) * norm(np_array_b))


@njit
def mean(l):
    return sum(l) / len(l)


@njit
def metric_MIFTS(MIFTS):
    a = 7
    k = 2

    if (MIFTS <= 75):
        return (k / (a - 1)) * (a ** ((100 - MIFTS) / 100) - 1)
    else:
        return (-1 / 300) * (MIFTS - 75) + 0.2089


@njit
def std(l, mean):
    if len(l) == 1:
        return 0
    else:
        return math.sqrt(sum([(i - mean) ** 2 for i in l]) / (len(l) - 1))


@njit
def similarity(list_symptoms1, list_symptoms2):
    if len(list_symptoms1) < len(list_symptoms2):
        a = list_symptoms1
        b = list_symptoms2
    else:
        a = list_symptoms2
        b = list_symptoms1

    cos_similarties_a = []

    for a_i in prange(len(a)):
        array_a = a[a_i]

        max_cos_sim_a = 0

        for b_i in prange(len(b)):
            array_b = b[b_i]

            cos_similarity = cosine_similarity(array_a, array_b)
            if cos_similarity > max_cos_sim_a:
                max_cos_sim_a = cos_similarity

        cos_similarties_a.append(max_cos_sim_a)

    return cos_similarties_a


def get_disease(symptoms_emb_nparr, npzfile, MIFTS_disease):
    disease_sims = {}

    getting_symptoms_time = 0
    similairty_time = 0

    for disease_index in range(len(npzfile.keys())):
        disease = list(npzfile.keys())[disease_index]

        MIFTS = MIFTS_disease[disease]

        disease_emb_nparr = npzfile[disease]

        cos_sims = similarity(symptoms_emb_nparr, disease_emb_nparr)

        mean_cos_sim = mean(cos_sims)
        std_cos_sim = std(cos_sims, mean_cos_sim)

        # Number of disease symptoms that don't link up to the user's symptoms
        num_nomatchsymps = abs(len(disease_emb_nparr) - len(symptoms_emb_nparr))

        # More no match symptoms means that the metic increases so the disease is a worse diagnosis
        metric = (1 - mean_cos_sim) + (num_nomatchsymps / 6) + (std_cos_sim / 4) + metric_MIFTS(MIFTS)

        disease_sims[disease] = metric

    return dict(sorted(disease_sims.items(), key=lambda item: item[1]))


# Load gzipped json
def load(type_disease):
    with gzip.open(f"symptomsfiles/best_{type_disease}.json.gz", 'rt', encoding='utf-8') as f:
        data = json.load(f)

    return data


# Given a list of dicts where each dict has the disease as a key and the list of symptoms as its value
# Search/find the disease symptoms pair and return it
def find(data, disease):
    for disease_dict in data:
        if list(disease_dict.keys())[0] == disease:
            return [disease, disease_dict[disease]]


def predict_symptoms(symptoms, type_disease):
    symptoms_emb_nparr = np.array(embed(symptoms))

    npzfile = np.load(f"symptomsfiles/best_{type_disease}_numpy.npz")

    with open(f"symptomsfiles/{type_disease}_MIFTS.json", "r") as f:
        MIFTS_disease_json = json.load(f)

    top5 = list(get_disease(symptoms_emb_nparr, npzfile, MIFTS_disease_json).items())[:5]

    combined_symptoms = "".join(f"{symptom}. " for symptom in symptoms)
    top5_diseases = [t[0] for t in top5]

    top_5_zero_shot = classifier(combined_symptoms, top5_diseases, multi_label=True)

    top_5_dict = {top_5_zero_shot["labels"][i]: top_5_zero_shot["scores"][i] for i in range(5)}

    skin_json = load(type_disease)

    top5_diseases_symptoms = {}

    for disease in top_5_dict.keys():
        confidence = top_5_dict[disease]

        disease, symptoms = find(skin_json, disease)

        top5_diseases_symptoms[disease] = (symptoms, confidence)

    return top5_diseases_symptoms
