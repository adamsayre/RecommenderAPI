# RecommenderAPI

Recommender API for skinsearch.co, created in Summer 2021 for MIDS Capstone
Other collaborators: @dineshachuthan @xtinachen

My main contribution were the .py files in the utils folder. These generate recommendations through the flask app in app.py.

Instructions:
* from the RecommenderAPI folder, run `python utils/recommender2.py`
  * this takes the raw review data and does the SVD decomposition/recomposition
  * it also does the bag of words on the products data
  * it takes cosine similarity within each of the two matrices described above and places the resulting matrix in the data directory
* also from the RecommenderAPI folder, run `python app.py` to activate the Flask app
* add to the end of the Flask url any of the following to see the returned recommendation
  * `/WithIngredients/genus/Moisturizers/skin/combination?n_recs=5`
  * `/WithoutIngredients/genus/Moisturizers/skin/combination?n_recs=5`
  * `/RoutineWithIngredients/genus/Moisturizers/skin/sensitive`
  * each returns a list of SKU numbers corresponding to the products data in the data directory

This app was called by other components of SkinSearch. 
