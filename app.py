# Application for recommendations
# example calls:
#   ~ /WithIngredients/genus/Moisturizers/skin/combination?n_recs=5
#   ~ /WithoutIngredients/genus/Moisturizers/skin/combination?n_recs=5
#   ~ /RoutineWithIngredients/genus/Moisturizers/skin/sensitive

from flask import Flask, request
import pickle
from utils.inference import recommenderWithIngredients
from utils.inference import recommenderWithoutIngredients
from utils.routine import Routine
import pandas as pd
from pprint import pprint


valid_genus = {
    'Cleansers',
    'Accessories',
    'Moisturizers',
    'Eye Care',
    'Treatments',
    'Masks',
    'Skincare',
    'Wellness',
    'Bath & Shower',
    'Face',
    'Sun Care',
    'Body Moisturizers',
    'Lip Treatments',
    'Self Tanners',
    'Brushes & Applicators',
    'High Tech Tools',
    'Hair Styling & Treatments',
    'Eye',
    'Lip'
}

skin_type = {
    "dry",
    "normal",
    "oily",
    "combination",
    "sensitive"
}
time_periods = {"day","night"}

app = Flask(__name__)
product_info = 'data/products_info.csv'
collab_info = 'data/collab_info.csv'
bag_info = 'data/bag_info.csv'

products_up = pd.read_csv(product_info)
collab_up = pd.read_csv(collab_info)
bag_up = pd.read_csv(bag_info)


@app.route("/WithoutIngredients/genus/<genus>/skin/<sensitivity>", methods=["POST","GET"])
def check(genus, sensitivity):
    n_recs = request.args.get('n_recs', 5)
    if(int(n_recs)) > 5:
        n_recs = 5

    show_cos_sim = request.args.get('show_cos_sim', False)
    if(genus in valid_genus and sensitivity in skin_type):
        return recommenderWithoutIngredients(
            products=products_up,
            collab=collab_up,
            category=genus,
            skin_type=sensitivity,
            n_recs=int(n_recs),
            show_cos_sim=bool(show_cos_sim)
        )
    else:
        return {
            "Valid Genus": ['Cleansers', 'Accessories', 'Moisturizers', 'Eye Care', 'Treatments', 'Masks', 'Skincare',
                      'Wellness', 'Bath & Shower', 'Face', 'Sun Care', 'Body Moisturizers', 'Lip Treatments',
                      'Self Tanners', 'Brushes & Applicators', 'High Tech Tools', 'Hair Styling & Treatments', 'Eye',
                      'Lip'],
            "Valid Skin Types": ["oily", "dry", "sensitive", "combination", "normal", "mature"]
        }



@app.route("/WithIngredients/genus/<genus>/skin/<sensitivity>", methods=["POST", "GET"])
def checkWithIngredients(genus, sensitivity):
    n_recs = request.args.get('n_recs', 5)
    if(int(n_recs)) > 5:
        n_recs = 5

    show_cos_sim = request.args.get('show_cos_sim', False)
    if(genus in valid_genus and sensitivity in skin_type):
        return recommenderWithIngredients(
            products=products_up,
            collab=collab_up,
            bag= bag_up,
            category=genus,
            skin_type=sensitivity,
            n_recs=int(n_recs),
            show_cos_sim=bool(show_cos_sim)
        )
    else:
        return {
            "Valid genus": ['Cleansers', 'Accessories', 'Moisturizers', 'Eye Care', 'Treatments', 'Masks', 'Skincare',
                      'Wellness', 'Bath & Shower', 'Face', 'Sun Care', 'Body Moisturizers', 'Lip Treatments',
                      'Self Tanners', 'Brushes & Applicators', 'High Tech Tools', 'Hair Styling & Treatments', 'Eye',
                      'Lip'],
            "Valid Skin Types": ["oily", "dry", "sensitive", "combination", "normal", "mature"]
        }


@app.route("/RoutineWithIngredients/genus/<genus>/skin/<sensitivity>", methods=["POST", "GET"])
def checkRoutineWithIngredients(genus, sensitivity):
    """Checks inputs and extra arguments as needed"""
    if sensitivity in skin_type:
        user_data = {'skin_type': sensitivity}
        routine_requested = Routine(user_data)
        return routine_requested.create_routine()
    else:
        return {
            "Valid Skin Types": ["oily", "dry", "sensitive", "combination", "normal", "mature"]
        }
      
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)