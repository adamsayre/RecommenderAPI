import pandas as pd
import numpy as np


def prep_data(file_loc='data/'):
    # Import products data from csv
    products = pd.read_csv(file_loc + 'products_cleaned_final.csv')

    # Import bag data from csv
    bag_up = pd.read_csv(file_loc + 'bag_info.csv')

    # if there's an 'Unnamed: 0' column, then remove it
    # (should only need to remove the first time it's run)
    if 'Unnamed: 0' in bag_up.columns:
        bag_up.drop('Unnamed: 0', axis=1, inplace=True)

    # Import collab data from csv
    collab_up = pd.read_csv(file_loc + 'collab_info.csv')

    # if there's an 'Unnamed: 0' column, then remove it
    # (should only need to remove the first time it's run)
    if 'Unnamed: 0' in collab_up.columns:
        collab_up.drop('Unnamed: 0', axis=1, inplace=True)

    # Get ingredients interactions
    ingredient_interactions = pd.read_csv(file_loc + "ingredients_interactions.csv")

    # Checks
    # print("Shape of Products: {}".format(products.shape))
    # print("Shape of Bag: {}".format(bag_up.shape))
    # print("Shape of Collab: {}".format(collab_up.shape))
    # print("Shape of ingredient interactions: {}".format(ingredient_interactions.shape))

    return products, bag_up, collab_up, ingredient_interactions


class Routine:

    def __init__(self, user):
        """
        Inputs:
        user: dictionary of user data, contains keys for at least 'skin_type'
        time: string, either 'day' or 'night' to specify  which routine the user wants """
        self.user = user
        self.products, self.bag_up, self.collab_up, self.ingredient_interactions = prep_data()

        """All Species: 
        array(['Moisturizers', 'Mists & Essences', 'BB & CC Cream', 'Face Oils',
           'Night Creams', 'Setting Spray & Powder', 'Mini Size',
           'BB & CC Creams', 'Tinted Moisturizer', 'Decollete & Neck Creams',
           'Face Sets', 'Foundation', 'Face Wash & Cleansers', 'Exfoliators',
           'Toners', 'Makeup Removers', 'Face Wipes', 'Face Wash',
           'Blotting Papers', 'Body Lotions & Body Oils', 'Value & Gift Sets',
           'Scrub & Exfoliants', 'Facial Peels', 'Face Serums',
           'Blemish & Acne Treatments', 'Face Primer', 'Beauty Supplements',
           'Body Wash & Shower Gel', 'Anti-Aging', 'For Face',
           'Moisturizer & Treatments', 'Eye Creams & Treatments', 'Eye Masks',
           'Eye Cream', 'Eye Primer', 'Face Masks', 'Sheet Masks',
           'Lip Balm & Treatment', 'Lip Balms & Treatments', 'Face Sunscreen',
           'Body Sunscreen', 'Hand Cream & Foot Cream', 'Sunscreen',
           'Eyeshadow', 'Hair Oil', 'Concealer', 'Highlighter',
           'Sponges & Applicators', 'Color Correct'], dtype=object)"""

        # # FOR DAY ROUTINE # #

        # For day
        # 1. cleanser (species = 'Face Wash & Cleansers')
        # 2. toner (species = 'Toners')
        # 3. serum (species = 'Face Serums')
        # 4. eye cream (species in ['Eye Creams & Treatments','Eye Cream']
        #       todo: should the treatments one be in here? or just save for night?
        # 5. acne spot treatment (species == 'Blemish & Acne Treatments')
        # 6. moisturizer (species in ['Moisturizers', 'Moisturizer & Treatments')
        #       todo: excludes tinted, is that what we want?, also same question on treatments
        # 7. sunscreen (species in ['Face Sunscreen', 'Sunscreen')
        #       todo: should other species be in here? saw 'Body Sunscreen' but that seemed out of scope

        # For routine, individual items are all in lists so we can call "in" on them later
        self.day_routine = [['Face Wash & Cleansers'],
                            ['Toners'],
                            ['Face Serums'],
                            ['Eye Creams & Treatments', 'Eye Cream'],
                            ['Blemish & Acne Treatments'],
                            ['Moisturizers', 'Moisturizer & Treatments'],
                            ['Face Sunscreen', 'Sunscreen']]

        # # FOR NIGHT ROUTINE # #
        # 1. cleanser (species in ['Face Wash & Cleansers']
        # 2. toner (species in ['Toners']
        # 3. eye cream (species in ['Eye Creams & Treatments', 'Eye Cream'])
        # 4. treatments (retinol, acne, serum, peel) (species in ['Face Serums', 'Blemish & Acne Treatments']
        #       todo: should we add some other of the species with treatments in the name?
        #       todo: peel isn't sth i'll recommend since it's not a daily?
        #         maybe this focuses mostly on daily and not on once-in-a-while
        # 5. moisturizer or night cream ['Moisturizers', 'Moisturizer & Treatments', 'Night Cream']
        #       todo: should this just be night cream? idk, seems like we'll moisturizers

        self.night_routine = [['Face Wash & Cleansers'],
                              ['Toners'],
                              ['Eye Creams & Treatments', 'Eye Cream'],
                              ['Facial Peels', 'Face Serums', 'Blemish & Acne Treatments'],
                              ['Moisturizers', 'Moisturizer & Treatments', 'Night Cream']]

    def recommender(self, products, collab, bag, category, skin_type, n_recs=10, starting_sku=np.nan,
                    show_cos_sim=False):
        """Find the recommended product in a given category for a given user, using collab similarities and bag of words similarities
        :Arguments:
        products: dataframe of all products
        collab: dataframe of cosine similarity of each product using collab filtering method
        bag: dataframe of the cosine similarity of each product using bag of words method
        category: list of strings, matching the 'species' of product in the product dataset
        skin_type: string, with type of skin, types listed in error message
        n_recs: integer, number of recommendations requested, default 10
        starting_sku: integer, sku to start recommendation with if provided, default is NaN
        show_cos_sim: boolean, true to show the cos_sim in the final matrix for debugging, default False
        """

        skin_type = skin_type.lower()

        # todo: add error check to make sure nrecs is greater than zero

        # get the product data that matches the category and skin type arguments
        in_category = products[products['species'].isin(category)]

        in_category = in_category[in_category[skin_type] == True]

        # If a starting sku is presented then we should use that to begin our recommendations
        if starting_sku is np.nan:
            # If no starting sku is presented, then
            # Find the product in category with the most loves (this will let us have a starting point for the ranking)
            starting_df = in_category.sort_values(by='num_loves', ascending=False)[0:1]
            starting_product = starting_df.index
            starting_sku = int(starting_df['sku'].values)
        else:
            # otherwise, find the index for that sku, we look in all products here because the starting product may
            # not be in the same category as the new recommendation
            starting_product = products.loc[products['sku'] == starting_sku][0:1].index

        # print("Starting Product is: {}".format(starting_product))

        # # COLLABORATIVE FILTERING RANKING # #

        # use the index from the category to get the similaraity, and sort by similarity
        try:
            # Checking if the starting product sku is in the collab file, not all skus are present in the collab file
            collab_skus = [int(i) for i in collab.columns.values]

            # This will pass a value error if the starting product isn't in there
            collab_location = collab_skus.index(starting_sku)

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

        # use the indices from the category to get the similarity and sort by similarity
        bag_rank = bag.iloc[starting_product, :].rank(ascending=False, method='min', axis=1)

        # combine bag rank and sku's from the product data to make a ranking
        bag_df = pd.DataFrame({'sku': products['sku'], 'bag_rank': bag_rank.iloc[0].values})

        # # FINAL RANKING AND RESULTS # #

        # Some products aren't in the collab model, so we enclose the collab processing in a try
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

        # if the recs provided is less than n-recs then fill the rest with np.nan
        final_skus = products_final['sku'].to_list()
        while len(final_skus) < n_recs:
            final_skus.append(np.nan)

        return final_skus

    def get_recommendations(self, time):
        """Obtain a routine recommendation
        Inputs:
        time: string matching either "day" or "night" """

        if time == "day":
            self.routine_to_use = self.day_routine
        else:
            self.routine_to_use = self.night_routine

        recs = np.zeros((10,len(self.routine_to_use)))


        # Use recommender function to get first recommendation
        recs[:, 0] = self.recommender(products=self.products,
                                      bag=self.bag_up,
                                      collab=self.collab_up,
                                      category=self.routine_to_use[0],
                                      skin_type=self.user['skin_type'],
                                      n_recs=10)
        # print("Recs 1:")
        # print(recs)
        # print(" ")
        # Get second recommendation
        recs[:, 1] = self.recommender(products=self.products,
                                      bag=self.bag_up,
                                      collab=self.collab_up,
                                      category=self.routine_to_use[1],
                                      skin_type=self.user['skin_type'],
                                      n_recs=10,
                                      starting_sku=recs[0,0])
        counter = 2
        # print("Recs 2")
        # print(recs)
        # print(" ")
        # subsequent recommendations should be item to item recommendations and will use the sku from the previous reco
        for i in self.routine_to_use[2:]:
            # print("Recs {}:".format(counter))
            start_sku = recs[0, counter-1]
            # print("Starting SKU: {}".format(recs[0,counter-1]))
            recs[:, counter] = self.recommender(products=self.products,
                                                bag=self.bag_up,
                                                collab=self.collab_up,
                                                category=i,
                                                skin_type=self.user['skin_type'],
                                                n_recs=10,
                                                starting_sku=start_sku)
            counter += 1
            # print(recs)
            # print(" ")


        # print("Final recs:")
        return recs

    def check_ingredients(self, recs_matrix, current_recs = np.nan):
        """check ingredients in top recommended products, if bad interactions, then sub the 2nd recommended product
                inputs:
                recs_matrix = a matrix of integer SKUs for products recommended
                current_recs = a list of integer positions in the recs matrix, one position per column of matrix, NaN as default
                    used to provide secondary recommendations for recursion"""

        # check for current_recs, if not provided then take the first position in every matrix
        if current_recs is np.nan:
            # current recommendation is the first option out of each recommended product
            current_recs = [0 for i in range(recs_matrix.shape[1])]

        # get a list of skus from the recs_matrix using current_recs
        current_routine = recs_matrix[[current_recs], np.arange(len(current_recs))]

        # make a df so that current_routine is in the right order
        cr = pd.DataFrame({"sku":current_routine.tolist()[0]})

        # get a df of all products currently in routine
        current_products = self.products[self.products['sku'].isin(current_routine[0,:])][['sku', 'ingredients']]
        current_products = cr.merge(current_products, on='sku', how='left')

        # Create a holding place for some data in a new column
        new_column = []
        all_ing_in_routine = []
        for j in range(len(current_products.index)):
            current_index = current_products.index[j]
            text_to_add = []
            ingredients_list_to_check = str(current_products['ingredients'].values[j]).strip().split(', ')
            for i in self.ingredient_interactions['ing_a'].unique():
                if i in ingredients_list_to_check:
                    text_to_add.append(i)
                    all_ing_in_routine.append(i)
            new_column.append(text_to_add)

        current_products['ing_a'] = new_column

        # Setting a flag for whether we found a bad sku, if we find one we'll use it to break out of all of the loops
        # sorry there's a lotta loops here
        found_bad_sku = False
        for i in range(len(current_products.index)):
            if found_bad_sku: break
            # print("Checking Product: {}".format(current_products['sku'].values[i]))
            current_ingredients = current_products['ing_a'].values[i]
            for j in current_ingredients:
                if found_bad_sku: break
                potential_problems = self.ingredient_interactions[self.ingredient_interactions['ing_a'] == j]
                try:
                    potential_problems = potential_problems[potential_problems['pairs_well'] == False]
                except KeyError:
                    next
                for k in range(len(current_products['ing_a'].values)):
                    product_checking_against = current_products['ing_a'].values[k]
                    # print("Checking {} against {}".format(current_products['sku'].values[i],
                    #                                       current_products['sku'].values[k]))
                    if found_bad_sku:
                        break
                    for l in product_checking_against:
                        if found_bad_sku:
                            break
                        # print("Checking {} in {}, result is {}".format(l,
                        #                                                potential_problems['ing_b'].values,
                        #                                                l in potential_problems['ing_b'].values))
                        if l in potential_problems['ing_b'].values:
                            # we have a problem!
                            # need to find sku of product so we can replace it
                            problem_sku = current_products['sku'].values[k]
                            found_bad_sku = True
                            break
                        else:
                            # could we use next if continue does weird things
                            continue
        # print("Done, problem? {}".format(found_bad_sku))

        # if there are any then call the ingredient checker function again but with a different set of products
        if found_bad_sku is False:
            return current_routine
        else:
            # Recursion time! we know which sku caused the problem from before so we find where it is in the matrix
            bad_position = np.where(current_routine.flatten() == problem_sku)[0][0]

            # Add one to the current recs for that spot so the next product gets picked when we run interactions again
            current_recs[bad_position] += 1

            # after iterating to the next product for all bad products, we call the interactions checker again
            return self.check_ingredients(recs_matrix, current_recs)

    def create_routine(self):
        """master function to create a routine"""

        day_routine = self.get_recommendations("day")
        night_routine = self.get_recommendations("night")
        # print("Routine:")
        # print(routine)
        out_am = self.check_ingredients(day_routine).flatten().tolist()
        out_pm = self.check_ingredients(night_routine).flatten().tolist()

        return {"am": [int(i) for i in out_am],
                "pm": [int(i) for i in out_pm]}
