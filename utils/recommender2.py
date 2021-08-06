import pickle
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer


def clean_embedding_text(input_text):
    """cleans embedding text
    - Removes digits and replaces with 'DG'
    - replaces '&' with 'and'
    - Removes '\n' and other non ascii characters"""

    output_text = input_text.lower().replace("&", "and").replace('\n', " ")
    output_text = re.sub(" +", " ", output_text)
    output_text = re.sub("[0-9]", "DG", output_text)
    output_text = re.sub("[^a-z ]", "", output_text)

    return output_text


class cf:
    
    def __init__(self):
        self.products, self.user_product_review = self.prep_data()
    
    def prep_data(self, file_loc = 'data/'):
        products = pd.read_csv(file_loc + 'products_cleaned_final.csv') #[["sku", "name", "brand", "family", "genus", "species", "num_loves"]]
        reviews = pd.read_csv(file_loc + "sephora_reviews_some.csv")
        reviews.sort_values(by='sku', inplace=True)
        # reviews.review_rating = reviews.review_rating.apply(lambda x: int(x.split(" ")[0]) if isinstance(x, str) else 0)
        user_product_review = pd.pivot_table(reviews, index='user_profile', columns='sku', values='review_rating').fillna(0)
        # user_product_matrix = user_product.values
        return products, user_product_review
        
    def train(self):
        U, sigma, Vt = svds(self.user_product_review.values, k=40)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        item_based = pd.DataFrame(all_user_predicted_ratings, 
                                  columns=self.user_product_review.columns, 
                                  index = list(self.user_product_review.index)).transpose()
        sim_matrix = cosine_similarity(item_based)
        # self.collab = pd.DataFrame(sim_matrix, columns=item_based.index, index=item_based.index)
        self.collab = pd.DataFrame(sim_matrix, columns=item_based.index.values, index=item_based.index.values).reset_index(drop=True)

    def train_bag(self):
        # First i need to combine the text of the different columns i want into the text i want to use to make the embedding
        self.products['embedding_text'] = self.products['name'] + " " + self.products['brand'] + " " + self.products['about']

        # Next, cleaning the embedding text
        self.products['embedding_text'] = self.products.embedding_text.apply(clean_embedding_text)

        # Next we use countvectorizer to develop the bag of words
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.products['embedding_text'].to_numpy())

        # And last use cosine similarity to get the cos sim matrix of all products
        sim_matrix = cosine_similarity(X)

        # finally put the similarity matrix into a dataframe and label properly
        self.bag = pd.DataFrame(sim_matrix, columns=self.products.index.values, index=self.products.index.values).reset_index(drop=True)

    def savePKL(self, model_name, data_to_save):
        """
        Function used to save the model to pickle file.
        model_name: name of pickle file to be saved
        data_to_save: list of dataframe/weights to be saved in pickle file. this can be unpickled later for inference. 
        In this case, the dataframe to be pickled are products (self.products) and similarity_matrix (self.collab), which are used for inference
        """
        with open(model_name,"wb") as f:
            for data in data_to_save:
                pickle.dump(data, f)

    def saveCSV(self, model_name_csv, data_to_save):
        with open(model_name_csv, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the data
            writer.writerow(data_to_save)


def main():
    # Now we run the function. God speed....
    import argparse
    import sys

    cf_recom = cf()
    cf_recom.train()
    cf_recom.train_bag()
    
    # save the model
    #model_name = 'weights3.pkl'

    #cf_recom.savePKL(model_name, [cf_recom.products, cf_recom.collab, cf_recom.bag])
    # cf_recom.saveCSV("../data/collab_info.csv", [cf_recom.collab])
    # cf_recom.saveCSV("../data/bag_info.csv", [cf_recom.bag])

    cf_recom.collab.to_csv('data/collab_info.csv')
    cf_recom.bag.to_csv('data/bag_info.csv')


if __name__ == "__main__":
    main()
