import math
import traceback

import numpy as np
import pandas as pd
import scipy
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# show all columns
pd.set_option('display.max_columns', None)


def smooth_user_preference(x):
    return math.log(1 + x, 2)


def get_rec(my_profile, num_of_rec):
    # load csv file
    df = pd.read_csv("data/pre_processed.csv")

    # _id_x,customerId,productId,interactionType,productName,description,productCatrgoryName,productCategoryId
    df.rename(columns={
        'orderedBy': 'user_id'
    }, inplace=True)

    df = df.drop_duplicates()
    df = df.dropna()

    #############################################################################################

    interactions_full_df = df.groupby(['user_id', 'food_id'])['feedback'].sum().apply(
        smooth_user_preference).reset_index()

    new_train_df, new_test_df = train_test_split(
        interactions_full_df,
        stratify=interactions_full_df['user_id'],
        test_size=0.20,
        random_state=42
    )

    # Indexing by user_id to speed up the searches during evaluation
    full_indexed_df = interactions_full_df.set_index('user_id')

    # Computes the most popular items
    def most_popular_items(interactions_df, item_id, topn=10):
        # Get a count of user_ids for each unique food as
        # a series of food ids and popularity
        item_popularity_df = interactions_df.groupby(item_id)['feedback'].sum().sort_values(
            ascending=False).reset_index()
        return item_popularity_df

    try:
        class PopularityRecommender:
            MODEL_NAME = 'Popularity'

            def __init__(self, popularity_df, items_df=None):
                self.popularity_df = popularity_df
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                recommendations_df = self.popularity_df[~self.popularity_df['food_id'].isin(items_to_ignore)] \
                    .sort_values('feedback', ascending=False) \
                    .head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='food_id',
                                                                  right_on='food_id')[
                        ['feedback', 'food_id']]

                return recommendations_df

        popularity_model = PopularityRecommender(most_popular_items(interactions_full_df, 'food_id'))

        popularity_flag = True

    # on error flag is set to false
    except Exception as e:
        print(traceback.format_exc())
        popularity_flag = False


    ####################################################################### Content-Based Filtering model

    try:
        # Ignoring stopwords (words with no semantics) from English language
        stopwords_list = stopwords.words('english')

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus,
        # ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)

        item_ids = df['food_id'].tolist()
        tfidf_matrix = vectorizer.fit_transform(df['food_name'] + "" + df['description'] + "" + df['cuisine'])
        tfidf_feature_names = vectorizer.get_feature_names_out()

        def get_item_profile(item_id):
            idx = item_ids.index(item_id)
            item_profile = tfidf_matrix[idx:idx + 1]
            return item_profile

        def get_item_profiles(ids):
            item_profiles_list = [get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def build_users_profile(person_id, interactions_indexed_df):
            interactions_person_df = interactions_indexed_df.loc[person_id]
            user_item_profiles = get_item_profiles(interactions_person_df['food_id'])

            user_item_strengths = np.array(interactions_person_df['feedback']).reshape(-1, 1)
            # Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
                user_item_strengths)

            user_item_strengths = np.array(interactions_person_df['feedback']).reshape(-1, 1)
            # Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
                user_item_strengths)
            user_item_strengths_weighted_avg = np.asarray(user_item_strengths_weighted_avg)
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            return user_profile_norm

        def build_users_profiles():
            interactions_indexed_df = new_train_df[
                new_train_df['food_id'].isin(df['food_id'])].set_index('user_id')
            user_profiles = {}
            for person_id in interactions_indexed_df.index.unique():
                user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
            return user_profiles

        print("Building user profiles...")
        user_profiles = build_users_profiles()

        class ContentBasedRecommender:
            MODEL_NAME = 'Content-Based'

            def __init__(self, items_df=None):
                self.item_ids = item_ids
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def _get_similar_items_to_user_profile(self, person_id, topn=1000):
                # Computes the cosine similarity between the user profile and all item profiles
                cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
                # Gets the top similar items
                similar_indices = cosine_similarities.argsort().flatten()[-topn:]
                # Sort the similar items by similarity
                similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                                       key=lambda x: -x[1])
                return similar_items

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                similar_items = self._get_similar_items_to_user_profile(user_id)
                # Ignores items the user has already interacted
                similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

                recommendations_df = pd.DataFrame(similar_items_filtered, columns=['food_id', 'recStrength'])
                # remove duplicates
                recommendations_df = recommendations_df.drop_duplicates(subset=['food_id'])
                # sort by recStrength
                recommendations_df = recommendations_df.sort_values(by='recStrength', ascending=False)
                # topn
                recommendations_df = recommendations_df.head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='food_id',
                                                                  right_on='food_id')[
                        ['recStrength', 'food_id', 'food_name']]

                return recommendations_df.drop_duplicates(subset=['food_id'])

        content_based_recommender_model = ContentBasedRecommender(df)

        cb_flag = True

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        cb_flag = False

    ####################################################################### Collaborative Filtering model

    try:
        # Creating a sparse pivot table with users in rows and items in columns
        users_items_pivot_matrix_df = new_train_df.pivot(index='user_id',
                                                         columns='food_id',
                                                         values='feedback').fillna(0)

        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_ids = list(users_items_pivot_matrix_df.index)
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

        # The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 15
        # Performs matrix factorization of the original user item matrix
        # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

        # Converting the reconstructed matrix back to a Pandas dataframe
        cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                                   index=users_ids).transpose()

        class CFRecommender:
            MODEL_NAME = 'Collaborative Filtering'

            def __init__(self, cf_predictions_df, items_df=None):
                self.cf_predictions_df = cf_predictions_df
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                # Get and sort the user's predictions
                sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                    .reset_index().rename(columns={user_id: 'recStrength'})

                # Recommend the highest predicted rating movies that the user hasn't seen yet.
                recommendations_df = sorted_user_predictions[~sorted_user_predictions['food_id'].isin(items_to_ignore)] \
                    .sort_values('recStrength', ascending=False) \
                    .head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='food_id',
                                                                  right_on='food_id')[
                        ['recStrength', 'food_id', 'food_name']]

                return recommendations_df.drop_duplicates(subset=['food_id'])

        cf_recommender_model = CFRecommender(cf_preds_df, df)

        cf_flag = True

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        cf_flag = False


    ####################################################################### Hybrid model

    try:
        class HybridRecommender:
            MODEL_NAME = 'Hybrid'

            def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
                self.cb_rec_model = cb_rec_model
                self.cf_rec_model = cf_rec_model
                self.cb_ensemble_weight = cb_ensemble_weight
                self.cf_ensemble_weight = cf_ensemble_weight
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                # Getting the top-1000 Content-based filtering recommendations
                cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                               topn=1000).rename(columns={'recStrength': 'recStrengthCB'})

                # Getting the top-1000 Collaborative filtering recommendations
                cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                               topn=1000).rename(columns={'recStrength': 'recStrengthCF'})

                # Combining the results by contentId
                recs_df = cb_recs_df.merge(cf_recs_df,
                                           how='outer',
                                           left_on='food_id',
                                           right_on='food_id').fillna(0.0)

                # Computing a hybrid recommendation score based on CF and CB scores
                # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
                recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (
                        recs_df['recStrengthCF'] * self.cf_ensemble_weight)

                # Sorting recommendations by hybrid score
                recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False)
                recommendations_df = recommendations_df.drop(['recStrengthCB', 'recStrengthCF'], axis=1)

                # Recommend the highest predicted rating movies that the user hasn't seen yet.
                recommendations_df = recommendations_df[~recommendations_df['food_id'].isin(items_to_ignore)] \
                    .sort_values('recStrengthHybrid', ascending=False) \
                    .head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='food_id',
                                                                  right_on='food_id')[
                        ['recStrengthHybrid', 'food_id', 'food_name']]

                return recommendations_df.drop_duplicates(subset=['food_id']).reset_index(drop=True)

        hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, df,
                                                     cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

        hybrid_flag = True

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        hybrid_flag = False


    print("User profiles:")
    print(user_profiles)
    if my_profile in user_profiles:
        myprofile = user_profiles[my_profile]

        pd.DataFrame(sorted(zip(tfidf_feature_names,
                                myprofile.flatten().tolist()), key=lambda x: -x[1])[:20],
                     columns=['token', 'relevance'])

        try:
            if hybrid_flag:
                recommadation = hybrid_recommender_model.recommend_items(my_profile, topn=10, verbose=True)
            elif cf_flag:
                recommadation = cf_recommender_model.recommend_items(my_profile, topn=10, verbose=True)
            elif cb_flag:
                recommadation = content_based_recommender_model.recommend_items(my_profile, topn=10, verbose=True)
            else:
                recommadation = popularity_model.recommend_items(my_profile, topn=10, verbose=True)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            recommadation = popularity_model.recommend_items(my_profile, topn=10, verbose=False)

        print("User profile found")
        print("popularity_flag", popularity_flag)
        print("cf_flag", cf_flag)
        print("cb_flag", cb_flag)
        print("hybrid_flag", hybrid_flag)

        return recommadation.head(num_of_rec)['food_id'].tolist(), 1

    else:
        # return top 10 popular items
        item_popularity_df = popularity_model.recommend_items(my_profile, topn=10, verbose=False)
        item_popularity_df_head = item_popularity_df.head(num_of_rec)['food_id'].tolist()

        print("User profile not found")
        print("popularity_flag", popularity_flag)
        print("cf_flag", cf_flag)
        print("cb_flag", cb_flag)
        print("hybrid_flag", hybrid_flag)

        return item_popularity_df_head, 0


# print(get_rec("64315d86362c27c707fe155c", 5))

# motor==3.1.2
# pymongo==4.3.3
