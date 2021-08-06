import numpy as np
import json
import pandas as pd


def recommenderWithoutIngredients(products, collab, category, skin_type, n_recs=10, show_cos_sim=False):
    """Find the recommended product in a given category for a given user, using only collaborative similarities
    :Arguments:
    products: dataframe of all products
    collab: dataframe of cosine similarity of each product using collab filtering method
    category: string, with a group matching the 'genus' of product in the product dataset, types listed in error code
    skin_type: string, with type of skin, types listed in error message
    n_recs: integer, number of recommendations requested, default 10
    show_cos_sim: boolean, true to show the cos_sim in the final matrix for debugging, default False
    """
    # Error handling, skin type must be lower case
    skin_type = skin_type.lower()

    # todo: add error check to make sure nrecs is greater than zero

    # get the product data that matches the category and skin type arguments
    in_category = products[products.genus == category]

    in_category = in_category[in_category[skin_type] == True]
    in_category_ind = in_category.index

    # Find the product with the most loves (this will let us have a starting point for the ranking)
    starting_df = in_category.sort_values(by='num_loves', ascending=False)[0:1]
    starting_product = starting_df.index
    starting_sku = int(starting_df['sku'].values)

    # if there's an 'Unnamed: 0' column, then remove it
    # (should only need to remove the first time it's run)
    if 'Unnamed: 0' in collab.columns:
        collab.drop('Unnamed: 0', axis=1, inplace=True)

    # Some products aren't in the collab model, so we enclose the collab processing in a try
    try:
        # Checking if the starting product sku is in the collab file, not all skus are present in the collab file
        collab_skus = [int(i) for i in collab.columns.values]

        # This will pass a value error if the starting sku isn't in there
        collab_location = int(collab_skus.index(starting_sku))

    except ValueError:
        # Some skus aren't included in the collaborative filtering model, so the other option must be used
        raise ValueError("The products needed to recommend are not in the collaborative filtering model, choose the 'withIngredients' option instead")

    # Ranking the products similarity relative to the starting product
    collab_rank = pd.DataFrame({"rank":collab.iloc[collab_location, :].rank(ascending=False, method='min').transpose()})

    # pull the skus from the products in collab_rank
    collab_rank['sku'] = [int(i) for i in collab_rank.index.values]

    #  Merge the data frames for in_category into collab and bag (left joins)
    products_final = in_category.merge(collab_rank, on='sku', how='left')

    # Then create a new column final_rank
    products_final['final_rank'] = products_final['rank']

    # Then sort the values by final_rank and return the requested number of recommendations
    products_final = products_final.sort_values(by='final_rank')[0:n_recs]

    # # if show_cos_sim is on, obtain teh sims
    # ## Todo: show_cos_sim debugging arguments

    out = {'sku': products_final['sku'].to_list()}
    return json.dumps(out)


def recommenderWithIngredients(products, collab, bag, category, skin_type, n_recs=10, show_cos_sim=False):
    """Find the recommended product in a given category for a given user, using collab similarities and bag of words similarities
    :Arguments:
    products: dataframe of all products
    collab: dataframe of cosine similarity of each product using collab filtering method
    bag: dataframe of the cosine similarity of each product using bag of words method
    category: string, with a group matching the 'genus' of product in the product dataset, types listed in error code
    skin_type: string, with type of skin, types listed in error message
    n_recs: integer, number of recommendations requested, default 10
    show_cos_sim: boolean, true to show the cos_sim in the final matrix for debugging, default False
    """
    # Error handling, skin type must be lowercase
    skin_type = skin_type.lower()

    # todo: add error check to make sure nrecs is greater than zero

    # get the product data that matches the category and skin type arguments
    in_category = products[products.genus == category]

    in_category = in_category[in_category[skin_type] == True]
    in_category_ind = in_category.index

    # Find the product with the most loves (this will let us have a starting point for the ranking)
    starting_product = in_category.sort_values(by='num_loves', ascending=False)[0:1].index

    # print("Starting Product is: {}".format(starting_product))

    # # COLLABORATIVE FILTERING RANKING # #

    # if there's an 'Unnamed: 0' column, then remove it
    # (should only need to remove the first time it's run)
    if 'Unnamed: 0' in collab.columns:
        collab.drop('Unnamed: 0', axis=1, inplace=True)

    # Some products aren't in the collab model, so we enclose the collab processing in a try
    try:
        # Checking if the starting product sku is in the collab file, not all skus are present in the collab file
        collab_skus = [int(i) for i in collab.columns.values]

        # This will pass a value error if the starting product isn't in there
        collab_location = collab_skus.index(starting_product)

        # Using this variable to exclude collab ranking later on
        in_collab = True

        # Ranking the products similarity relative to the starting product
        collab_rank = collab.iloc[collab_location, :].rank(ascending=False, method='min', axis=1).transpose()

        # pull the skus from the products in collab_rank
        collab_rank['sku'] = [int(i) for i in collab_rank.index.values]
    except ValueError:
        # Set this flag to exclude operations later
        in_collab = False

    # # BAG OF WORDS RANKING # #

    # if there's an 'Unnamed: 0' column, then remove it
    # (should only need to remove the first time it's run)
    if 'Unnamed: 0' in bag.columns:
        bag.drop('Unnamed: 0', axis=1, inplace=True)

    # use the indices from teh category to get the similarity and sort by similarity
    bag_rank = bag.iloc[starting_product, :].rank(ascending=False, method='min', axis=1)

    # combine bag rank and sku's from the product data to make a ranking
    bag_df = pd.DataFrame({'sku': products['sku'], 'bag_rank': [int(i) for i in bag_rank]})

    ## FINAL RANKING AND RESULTS ##

    # If the product isn't in collab, we take the ranking from just the bag model
    if in_collab:
        #  Merge the data frames for in_category into collab and bag (left joins)
        products_final = in_category.merge(collab_rank, on='sku', how='left').merge(bag_df, on='sku', how='left')

        # Then create a new column final_rank which is the sum of the ranks from the two methods
        products_final['final_rank'] = products_final[starting_product[0]] + products_final['bag_rank']
    else:
        # Merge the in_category df into the bag df (left join)
        products_final = in_category.merge(bag_df, on='sku', how='left')

        # Then create a new column final_rank which is just the bag rank since the collab doesn't have that product
        products_final['final_rank'] = products_final['bag_rank']

    # Then sort the values by final_rank and return the requested number of recommendations
    products_final = products_final.sort_values(by='final_rank')[0:n_recs]

    # if show_cos_sim is on, obtain teh sims
    ## Todo: show_cos_sim debugging arguments

    out = {'sku': products_final['sku'].to_list()}
    return json.dumps(out)





def main():
    # Now we run the function. God speed....
    import argparse
    
    # Initialize parser
    parser = argparse.ArgumentParser(prog="Skincare Recommender")
    
    # Adding optional argument
    parser.add_argument('--cat',
                        choices=['Moisturizers', 'Masks', 'Cleansers', 'Treatments', 'Eye Care', 'Sun Care', 'Wellness', 'High Tech Tools', 'Self Tanners', 'Lip Treatments'],
                        required=True,
                        help="Category to use for recommendation")
    parser.add_argument('--skin', 
                        choices =["oily","dry","sensitive","combination","normal","mature"], 
                        required=True,
                       help="Skin Type to use for recommendation")
    parser.add_argument('--recs',
                        required=False, 
                        type=int, 
                        default=10, 
                        help="Number of recommendations requested (default: %(default)s)")
    parser.add_argument('--cos',
                        required=False, 
                        type=bool, 
                        default=False, 
                        help='Bool to display the cosine similarities for debugging (default: %(default)s)')
    parser.add_argument('--start',
                        required=False,
                        type=int,
                        default=np.nan,
                        help='int to use as starting product')
    
    args = parser.parse_args()
    # print(args.cat, args.skin)

    product_info = '../data/products_info.csv'
    collab_info = '../data/collab_info.csv'
    bag_info = '../data/bag_info.csv'

    products_up = pd.read_csv(product_info)
    collab_up = pd.read_csv(collab_info)
    bag_up = pd.read_csv(bag_info)

    recommendations = recommenderWithIngredients(products=products_up,
                                  collab=collab_up,
                                  bag=bag_up,
                                  category=args.cat,
                                  skin_type=args.skin,
                                  n_recs=args.recs,
                                  show_cos_sim=args.cos)

    print(f'Recommendations with ingredients is: {recommendations}')

    recommendations = recommenderWithoutIngredients(products=products_up,
                                  collab=collab_up,
                                  category=args.cat,
                                  skin_type=args.skin,
                                  n_recs=args.recs,
                                  show_cos_sim=args.cos)

    print(f'Recommendations without ingredients is: {recommendations}')


if __name__ == "__main__":
    main()
