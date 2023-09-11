import sys
import json
from pyspark import SparkContext, StorageLevel, SparkConf
from sklearn.preprocessing import LabelEncoder
from math import sqrt, isnan
from itertools import product
from pandas import DataFrame
from ast import literal_eval
import pandas as pd
from collections import defaultdict
from datetime import date, datetime
import pickle

'''
Method Description:
For this method I used a weighted hybrid recommender system model using a Model-Based Recommender and an Item-Based Recommender. The Model-Based Recommender used an XGBoostRegressor to predict ratings given a set of features, while the item-based recommender uses its neighbors and the pearson correlation metric to compute the ratings. These predictions are then combined using a weight value of alpha = 0.01, which I computed through trial and error, to create a final prediction. The parameters that I have hardcoded for the XGBRegressor were computed using hyperparameter tuning. This helped to greatly improve the RMSE score and adding the item-based cf to create a hybrid model also helped with the RMSE marginally. However, one of the main ways that I improved on RMSE was through suitable selection of features. I created close to 40 features from all of business.json, user.json, and tip.json. I extracted features from each and/or created new ones by combining multiple features or other such methods. Then using an exhaustive Recursive Feature Elimination algorithm offline, I was able to select the best features for the model. All of these methods helped to greatly increase the accuracy of my model. Since the XGBRegressor model is very time consuming, I used a pickle file to generate the training weights which I then use to make predictions here. I have also uploaded train.py which generates these training weights and a pickle file which contains these trained weights.

Error Distribution:
>=0 and <1: 102290
>=1 and <2: 32808
>=2 and <3: 6155
>=3 and <4: 791
>=4 and <5: 0

RMSE:
0.9781

Execution Time:
165s
'''

class Item_Based_CF():

    def __init__(self, co_rated_user_threshold, n_neighbors):
        self.co_rated_user_threshold = co_rated_user_threshold
        self.n_neighbors = n_neighbors

    def read_dataset(self, sc, filepath):
        return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

    def create_user_item_matrix(self, rdd):
        return rdd.map(lambda x: ((x[0], x[1]), float(x[2])))

    def get_item_averages(self, rdd):
        return rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()

    def get_item_review_count(self, rdd):
        return rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()

    def get_user_averages(self, rdd):
        return rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()

    def get_user_review_count(self, rdd):
        return rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()

    def _compute_pearson(self, user_item_matrix, relevant_users, item_pair, co_rated_averages):
        weight_num = 0
        weight_denom1 = 0
        weight_denom2 = 0
        for user in relevant_users:
            weight_num += (user_item_matrix[(user, item_pair[0])]-co_rated_averages[item_pair[0]]) \
                *(user_item_matrix[(user, item_pair[1])]-co_rated_averages[item_pair[1]])
            weight_denom1 += (user_item_matrix[(user, item_pair[0])]-co_rated_averages[item_pair[0]])**2
            weight_denom2 += (user_item_matrix[(user, item_pair[1])]-co_rated_averages[item_pair[1]])**2

        weight_denom = sqrt(weight_denom1)*sqrt(weight_denom2)
        if weight_denom == 0:
            return 0
        return weight_num/weight_denom

    def _shrink_item_set(self, unique_items, item_user_map, user_train_map, user_item_pair):
        res_items = set()
        for item in unique_items:
            if user_item_pair[1] in item_user_map and item != user_item_pair[1]:
                if len(item_user_map[user_item_pair[1]].intersection(item_user_map[item])) >= self.co_rated_user_threshold \
                    and (user_item_pair[0], item) in user_train_map:
                    res_items.add(item)

        if len(res_items) == 0:
            return [0]
        return res_items

    def _compute_co_rated_average(self, user_item_matrix, item_user_map, item_pair):
        relevant_users = item_user_map[item_pair[0]].intersection(item_user_map[item_pair[1]])
        res = {}
        for user_item_pair in product(relevant_users, item_pair):
            if user_item_pair[1] not in res:
                res[user_item_pair[1]] = []
            
            res[user_item_pair[1]].append(user_item_matrix[user_item_pair])
            
        return dict(map(lambda x: (x[0], sum(x[1])/len(x[1])), res.items()))

    def compute_weight(self, test_rdd, user_item_map_train, user_item_map, item_user_map):
        weight_combination_rdd = test_rdd.map(lambda x: ((x[0], x[1]), user_item_map[x[0]]) if x[0] in user_item_map else ((x[0], x[1]), set([]))) \
            .map(lambda x: (x[0], self._shrink_item_set(x[1], item_user_map, user_item_map_train, x[0]))).flatMapValues(lambda x: x) \
                .map(lambda x: (x[0], ((x[0][1], x[1]), item_user_map[x[0][1]].intersection(item_user_map[x[1]]), \
                    self._compute_co_rated_average(user_item_map_train, item_user_map, (x[0][1], x[1])))) if type(x[1]) == str else (x[0], 0)) \
                        .mapValues(lambda x: {x[0]: self._compute_pearson(user_item_map_train, x[1], x[0], x[2])} if type(x) == tuple else [0]) \
                            .groupByKey().mapValues(list) \
                                .mapValues(lambda x: {list(dic.items())[0][0]: dic[list(dic.items())[0][0]] for dic in x} if type(x[0]) == dict else [0])
        return weight_combination_rdd

    def predict(self, user_item_matrix, weights, user_item_pair, user_item_map, item_user_map, average_user_ratings, \
        average_item_ratings, overall_average_rating):
        try:
            assert (user_item_pair[0] in user_item_map) and (user_item_pair[1] in item_user_map)
            if type(weights) == list:
                return (average_user_ratings[user_item_pair[0]]+average_item_ratings[user_item_pair[1]])/2
            
            new_weights = {}
            for weight in weights:
                if weights[weight] != 0:
                    new_weights[weight]
            
            num1 = 0
            denom1 = 0
            neighbor_weights = dict(sorted(new_weights.items(), key=lambda x: x[1], reverse=True)[:self.n_neighbors])
            for weight in neighbor_weights:
                num1 += user_item_matrix[(user_item_pair[0], weight[1])]*neighbor_weights[weight]
                denom1 += abs(neighbor_weights[weight])

            if denom1 == 0:
                return (average_user_ratings[user_item_pair[0]]+average_item_ratings[user_item_pair[1]])/2
            return num1/denom1
        except Exception:
            if user_item_pair[0] not in user_item_map and user_item_pair[1] not in item_user_map:
                return overall_average_rating
            elif user_item_pair[0] in user_item_map and user_item_pair[1] not in item_user_map:
                return average_user_ratings[user_item_pair[0]]
            elif user_item_pair[0] not in user_item_map and user_item_pair[1] in item_user_map:
                return average_item_ratings[user_item_pair[1]]
            else:
                return overall_average_rating

class Dataset():

    def __init__(self, filename, review_json_filename, business_json_filename, user_json_filename, tip_json_filename):
        self.filename = filename
        self.review_json_filename = review_json_filename
        self.business_json_filename = business_json_filename
        self.user_json_filename = user_json_filename
        self.tip_json_filename = tip_json_filename

    def read_csv_dataset(self, sc, filepath):
        return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

    def read_json_dataset(self, sc, filepath):
        return sc.textFile(filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)

    def _flatten_tuple(self, tup, target=None, exclude=None, is_test=False):
        res = []
        if target and exclude:
            if is_test:
                res.append(target)
                for inner_obj in tup:
                    if type(inner_obj) != str:
                        for elem in inner_obj:
                            if elem != exclude:
                                res.append(elem)
            else:
                res.append(target)
                for inner_obj in tup:
                    for elem in inner_obj:
                        if elem != exclude:
                            res.append(elem)
        elif target and not exclude:
            res.append(target[0])
            res.append(target[1])
            for inner_obj in tup:
                for elem in inner_obj:
                    res.append(elem)
        else:
            for inner_obj in tup:
                if type(inner_obj) == tuple:
                    for elem in inner_obj:
                        res.append(elem)
                else:
                    res.append(inner_obj)

        return tuple(res)

    def _populate_empty_features(self, x, average_rounded_business_rating=None, average_latitude=None, average_longitude=None, average_user_rating=None, \
        average_num_user_reviews=None, average_num_compliments=None, average_num_reviews_per_year=None, biz=True, user=True, is_test=False):
        if biz:
            if is_test:
                return (x[0], average_rounded_business_rating, average_latitude, average_longitude)
            else:
                return (x[0], x[1][0][1], average_rounded_business_rating, average_latitude, average_longitude)
        if user:
            if is_test:
                return (x[0], average_user_rating, average_num_user_reviews, average_num_compliments, average_num_reviews_per_year)
            else:
                return (x[0], x[1][0][1], average_user_rating, average_num_user_reviews, average_num_compliments, average_num_reviews_per_year)

    def generate_metrics(self, sc):
        rdd = self.read_csv_dataset(sc, self.filename).map(lambda x: (x[0], 1)).collectAsMap()
        business_rdd = self.read_json_dataset(sc, self.business_json_filename)
        
        user_list = []
        with open(self.user_json_filename) as user_file:
            for line in user_file:
                line_dict = json.loads(line)
                if line_dict['user_id'] in rdd:
                    user_list.append(json.loads(line))

        user_df = pd.DataFrame(user_list)[['user_id', 'average_stars', 'review_count', 'yelping_since', \
            'compliment_cool', 'compliment_funny']]

        today = date(2022, 10, 28)
        user_df['avg_user_compliments'] = user_df[['compliment_cool', 'compliment_funny']].sum(axis=1) \
            .divide(user_df['review_count'], axis=0)
        user_df['avg_user_compliments'].fillna(0, inplace=True)
        user_df['yelping_since'] = user_df['yelping_since'].apply(lambda x: (today-datetime.strptime(x, '%Y-%m-%d').date()).days)
        min_user_duration = user_df['yelping_since'].min(axis=0)
        user_df['reviews_per_year'] = user_df['review_count'].multiply(365, axis=0) \
            .divide(user_df['yelping_since'].subtract(min_user_duration-1, axis=0), axis=0)

        rounded_business_rating_rdd = business_rdd.map(lambda x: (x['business_id'], x['stars']))
        business_latitude_rdd = business_rdd.map(lambda x: (x['business_id'], x['latitude']))
        business_longitude_rdd = business_rdd.map(lambda x: (x['business_id'], x['longitude']))
        business_metrics_rdd = rounded_business_rating_rdd.join(business_latitude_rdd).join(business_longitude_rdd) \
            .mapValues(lambda x: self._flatten_tuple(x))
        
        return business_metrics_rdd, user_df.drop(['yelping_since', 'compliment_cool', 'compliment_funny'], axis=1)

    def generate_dataset(self, sc, user_item_map, business_metrics_rdd, user_metrics_df, is_test=False):
        rdd = self.read_csv_dataset(sc, self.filename)
        
        business_metrics_map = business_metrics_rdd.collectAsMap()
        business_rdd = self.read_json_dataset(sc, self.business_json_filename)

        num_unique_businesses = business_rdd.count()
        average_rounded_business_rating = round(2*business_rdd.map(lambda x: x['stars']) \
            .filter(lambda x: True if isnan(float(x)) else False).sum()/num_unique_businesses)/2
        average_latitude = business_rdd.map(lambda x: x['latitude']).filter(lambda x: True if x else False).sum()/num_unique_businesses
        average_longitude = business_rdd.map(lambda x: x['longitude']).filter(lambda x: True if x else False).sum()/num_unique_businesses

        average_user_rating = user_metrics_df['average_stars'].mean()
        average_num_user_reviews = user_metrics_df['review_count'].mean()
        average_num_compliments = user_metrics_df['avg_user_compliments'].mean()
        average_num_reviews_per_year = user_metrics_df['reviews_per_year'].mean()

        user_metrics_rdd = sc.parallelize(user_metrics_df.values.tolist()).map(lambda x: (x[0], x[1:]))

        if is_test:
            empty_business_test_vals = rdd.map(lambda x: (x[1], x[0])).leftOuterJoin(business_metrics_rdd) \
                .filter(lambda x: True if x[0] not in business_metrics_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_rounded_business_rating=average_rounded_business_rating, \
                        average_latitude=average_latitude, average_longitude=average_longitude, user=False, is_test=is_test)))
            business_metrics_combined_rdd = rdd.map(lambda x: (x[1], x[0])).join(business_metrics_rdd) \
                .map(lambda x: (x[1][0], self._flatten_tuple(x[1], x[0], x[1][0], is_test=is_test))).union(empty_business_test_vals) \
                    .map(lambda x: ((x[0], x[1][0]), tuple(val for val in x[1][1:])))
            
            empty_user_test_vals = rdd.map(lambda x: (x[0], x[1])).leftOuterJoin(user_metrics_rdd) \
                .filter(lambda x: True if x[0] not in user_item_map else False) \
                    .map(lambda x: (x[1][0], self._populate_empty_features(x, average_user_rating=average_user_rating, \
                        average_num_user_reviews=average_num_user_reviews, average_num_compliments=average_num_compliments, \
                            average_num_reviews_per_year=average_num_reviews_per_year, biz=False, is_test=is_test)))
            user_metrics_combined_rdd = rdd.map(lambda x: (x[0], x[1])).join(user_metrics_rdd) \
                .map(lambda x: (x[1][0], self._flatten_tuple(x[1], x[0], x[1][0], is_test=is_test))).union(empty_user_test_vals) \
                    .map(lambda x: ((x[1][0], x[0]), tuple(val for val in x[1][1:])))
        else:
            empty_business_test_vals = rdd.map(lambda x: (x[1], (x[0], float(x[2])))).leftOuterJoin(business_metrics_rdd) \
                .filter(lambda x: True if x[0] not in business_metrics_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_rounded_business_rating=average_rounded_business_rating, \
                        average_latitude=average_latitude, average_longitude=average_longitude, user=False)))
            business_metrics_combined_rdd = rdd.map(lambda x: (x[1], (x[0], float(x[2])))).join(business_metrics_rdd) \
                .map(lambda x: (x[1][0][0], self._flatten_tuple(x[1], x[0], x[1][0][0]))).union(empty_business_test_vals) \
                    .map(lambda x: ((x[0], x[1][0]), tuple(val for val in x[1][1:])))

            empty_user_test_vals = rdd.map(lambda x: (x[0], (x[1], float(x[2])))).leftOuterJoin(user_metrics_rdd) \
                .filter(lambda x: True if x[0] not in user_item_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_user_rating=average_user_rating, \
                        average_num_user_reviews=average_num_user_reviews, average_num_compliments=average_num_compliments, \
                            average_num_reviews_per_year=average_num_reviews_per_year, biz=False)))
            user_metrics_combined_rdd = rdd.map(lambda x: (x[0], (x[1], float(x[2])))).join(user_metrics_rdd) \
                .map(lambda x: (x[1][0][0], self._flatten_tuple(x[1], x[0], x[1][0][0], is_test=is_test))).union(empty_user_test_vals) \
                    .map(lambda x: ((x[1][0], x[0]), tuple(val for val in x[1][2:])))
        
        if is_test:
            combined_rdd = business_metrics_combined_rdd.join(user_metrics_combined_rdd) \
                .map(lambda x: list(self._flatten_tuple(x[1], x[0])))
            column_names = ['user_id', 'business_id', 'rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', \
                'num_user_reviews', 'avg_user_compliments', 'reviews_per_year']
        else:
            combined_rdd = business_metrics_combined_rdd.join(user_metrics_combined_rdd) \
                .map(lambda x: list(self._flatten_tuple(x[1], x[0])))
            column_names = ['user_id', 'business_id', 'rating', 'rounded_business_rating', 'latitude', 'longitude', \
                'average_user_rating', 'num_user_reviews', 'avg_user_compliments', 'reviews_per_year']
        
        feature_matrix_as_list = combined_rdd.collect()
        feature_matrix = DataFrame(feature_matrix_as_list, columns=column_names)

        business_list = []

        with open(self.business_json_filename) as business_file:
            for line in business_file:
                business_list.append(json.loads(line))

        business_df = pd.DataFrame(business_list)
        new_feature_matrix = pd.merge(feature_matrix, business_df[['business_id', 'city', 'state', 'name', 'attributes', \
            'review_count']], on='business_id')
        new_feature_matrix['city_state'] = new_feature_matrix['city']+'-'+new_feature_matrix['state']
        new_feature_matrix['user_business_pair'] = tuple(zip(new_feature_matrix['user_id'], new_feature_matrix['business_id']))

        new_feature_matrix['price_range'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(x['RestaurantsPriceRange2']) if 'RestaurantsPriceRange2' in x else 0)
        new_feature_matrix['bike_parking'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['BikeParking'])) if 'BikeParking' in x else 0)
        new_feature_matrix['good_for_kids'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['GoodForKids'])) if 'GoodForKids' in x else 0)
        new_feature_matrix['has_TV'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['HasTV'])) if 'HasTV' in x else 0)
        new_feature_matrix['outdoor_seating'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['OutdoorSeating'])) if 'OutdoorSeating' in x else 0)
        new_feature_matrix['restaurants_delivery'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['RestaurantsDelivery'])) if 'RestaurantsDelivery' in x else 0)
        new_feature_matrix['restaurants_good_for_groups'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['RestaurantsGoodForGroups'])) if 'RestaurantsGoodForGroups' in x else 0)
        new_feature_matrix['restaurants_reservations'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['RestaurantsReservations'])) if 'RestaurantsReservations' in x else 0)
        new_feature_matrix['restaurants_take_out'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['RestaurantsTakeOut'])) if 'RestaurantsTakeOut' in x else 0)
        new_feature_matrix['by_appointment_only'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['ByAppointmentOnly'])) if 'ByAppointmentOnly' in x else 0)
        new_feature_matrix['wheelchair_accessible'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['WheelchairAccessible'])) if 'WheelchairAccessible' in x else 0)
        new_feature_matrix['open_24_hours'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['Open24Hours'])) if 'Open24Hours' in x else 0)
        new_feature_matrix['drive_thru'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['DriveThru'])) if 'DriveThru' in x else 0)
        new_feature_matrix['coat_check'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['CoatCheck'])) if 'CoatCheck' in x else 0)
        new_feature_matrix['restaurants_table_service'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['RestaurantsTableService'])) if 'RestaurantsTableService' in x else 0)
        new_feature_matrix['business_accepts_bitcoin'] = new_feature_matrix['attributes'].apply(lambda x: [] if x is None else x) \
            .apply(lambda x: int(bool(x['BusinessAcceptsBitcoin'])) if 'BusinessAcceptsBitcoin' in x else 0)

        new_feature_matrix['garage'] = new_feature_matrix['attributes'].apply(lambda x: {} if x is None else x) \
            .apply(lambda x: literal_eval(x['BusinessParking']) if 'BusinessParking' in x else {}) \
                .apply(lambda x: int(x['garage']) if 'garage' in x else 0)
        new_feature_matrix['lot'] = new_feature_matrix['attributes'].apply(lambda x: {} if x is None else x) \
            .apply(lambda x: literal_eval(x['BusinessParking']) if 'BusinessParking' in x else {}) \
                .apply(lambda x: int(x['lot']) if 'lot' in x else 0)

        new_feature_matrix['lunch'] = new_feature_matrix['attributes'].apply(lambda x: {} if x is None else x) \
            .apply(lambda x: literal_eval(x['GoodForMeal']) if 'GoodForMeal' in x else {}) \
                .apply(lambda x: int(x['lunch']) if 'lunch' in x else 0)
        
        new_feature_matrix['casual'] = new_feature_matrix['attributes'].apply(lambda x: {} if x is None else x) \
            .apply(lambda x: literal_eval(x['Ambience']) if 'Ambience' in x else {}) \
                .apply(lambda x: int(x['casual']) if 'casual' in x else 0)

        all_city_states = business_df['city']+'-'+business_df['state']
        unique_city_states = list(set(all_city_states.values))

        unique_names = list(set(business_df['name'].values))

        le_cs = LabelEncoder()
        le_cs.fit(unique_city_states)

        le_n = LabelEncoder()
        le_n.fit(unique_names)

        new_feature_matrix['city_state_le'] = le_cs.transform(new_feature_matrix['city_state'])
        new_feature_matrix['name_le'] = le_n.transform(new_feature_matrix['name'])

        line_tip_ub_pairs = defaultdict(list)
        with open(self.tip_json_filename) as tip_file:
            for line in tip_file:
                for line in tip_file:
                    ub_pair = json.loads(line)
                    line_tip_ub_pairs[(ub_pair['user_id'], ub_pair['business_id'])].append(ub_pair['likes'])

        tip_ub_pairs = [{ub_pair[0]: ub_pair[1]} for ub_pair in line_tip_ub_pairs.items()]
        tip_ub_pairs_list = [{'user_id': list(line.keys())[0][0], 'business_id': list(line.keys())[0][1], \
            'avg_ub_pair_likes': sum(list(line.values())[0])/len(list(line.values())[0])} for line in tip_ub_pairs]
        ub_pair_tip_df = pd.DataFrame(tip_ub_pairs_list)

        tip_user_likes = self.read_json_dataset(sc, self.tip_json_filename).map(lambda x: (x['user_id'], x['likes'])) \
            .groupByKey().mapValues(lambda x: (sum(x), sum(x)/len(x))).map(lambda x: [x[0], x[1][0], x[1][1]]) \
                .collect()
        tip_business_likes = self.read_json_dataset(sc, self.tip_json_filename).map(lambda x: (x['business_id'], x['likes'])) \
            .groupByKey().mapValues(lambda x: (max(x), sum(x), sum(x)/len(x))).map(lambda x: [x[0], x[1][0], x[1][1], x[1][2]]) \
                .collect()
        
        user_tip_df = pd.DataFrame(tip_user_likes, columns=['user_id', 'total_user_likes', 'avg_user_likes'])
        business_tip_df = pd.DataFrame(tip_business_likes, columns=['business_id', 'max_business_likes', 'total_business_likes', \
            'avg_business_likes'])

        new_feature_matrix = pd.merge(new_feature_matrix, user_tip_df, on='user_id', how='left')
        new_feature_matrix = pd.merge(new_feature_matrix, business_tip_df, on='business_id', how='left')
        new_feature_matrix = pd.merge(new_feature_matrix, ub_pair_tip_df, on=['user_id', 'business_id'], how='left').fillna(0)

        return new_feature_matrix

if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")

    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel('ERROR')

    train_file_name = folder_path + 'yelp_train.csv'
    pickle_file_name = 'xgb_reg.pkl'

    ##### Training Phase #####
    ### ITEM-BASED-CF MODEL ###

    item_based_cf_model = Item_Based_CF(5, 4)
    train_rdd = item_based_cf_model.read_dataset(sc, train_file_name)

    average_user_ratings = item_based_cf_model.get_user_averages(train_rdd)
    average_item_ratings = item_based_cf_model.get_item_averages(train_rdd)

    num_user_ratings = item_based_cf_model.get_user_review_count(train_rdd)
    num_item_ratings = item_based_cf_model.get_item_review_count(train_rdd)

    user_item_rdd_train = item_based_cf_model.create_user_item_matrix(train_rdd)
    user_item_map_train = user_item_rdd_train.collectAsMap()

    item_user_map = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    user_item_map = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    overall_average_rating = train_rdd.map(lambda x: float(x[2])).sum()/train_rdd.count()

    ### Generate Metrics ###

    review_json_filename = folder_path + 'review_train.json'
    business_json_filename = folder_path + 'business.json'
    user_json_filename = folder_path + 'user.json'
    tip_json_filename = folder_path + 'tip.json'

    train_dataset = Dataset(train_file_name, review_json_filename, business_json_filename, user_json_filename, tip_json_filename)
    business_metrics_rdd, user_metrics_rdd = train_dataset.generate_metrics(sc)

    ##### Testing Phase #####
    ### ITEM-BASED-CF MODEL ###

    test_rdd = item_based_cf_model.read_dataset(sc, test_file_name)
    user_item_map_test = test_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    weight_test_rdd = item_based_cf_model.compute_weight(test_rdd, user_item_map_train, user_item_map, item_user_map)
    prediction_rdd_item_based_test = weight_test_rdd \
        .map(lambda x: (x[0], item_based_cf_model.predict(user_item_map_train, x[1], x[0], user_item_map, \
            item_user_map, average_user_ratings, average_item_ratings, overall_average_rating)))
    
    ### MODEL-BASED RS ###

    xgb = pickle.load(open(pickle_file_name, "rb"))

    test_dataset = Dataset(test_file_name, review_json_filename, business_json_filename, user_json_filename, tip_json_filename)
    test_feature_matrix = test_dataset.generate_dataset(sc, user_item_map_test, business_metrics_rdd, user_metrics_rdd, True)

    X_test = test_feature_matrix[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', \
        'avg_user_compliments', 'reviews_per_year', 'review_count', 'price_range', 'bike_parking', 'good_for_kids', 'has_TV', \
            'outdoor_seating', 'restaurants_delivery', 'restaurants_good_for_groups', 'restaurants_reservations', \
                'restaurants_take_out', 'by_appointment_only', 'wheelchair_accessible', 'open_24_hours', 'drive_thru', \
                    'coat_check', 'restaurants_table_service', 'business_accepts_bitcoin', 'garage', 'lot', 'lunch', 'casual', \
                        'city_state_le', 'name_le', 'total_user_likes', 'avg_user_likes', 'max_business_likes', \
                            'total_business_likes', 'avg_business_likes', 'avg_ub_pair_likes']]
    y_pred_test = list(xgb.predict(X_test))

    test_features_list = test_feature_matrix[['user_id', 'business_id']].values.tolist()
    
    test_set_with_predictions = []
    for i, feats in enumerate(test_features_list):
        test_set_with_predictions.append(feats.copy())
        test_set_with_predictions[-1].append(y_pred_test[i])

    prediction_rdd_model_based_test = sc.parallelize(test_set_with_predictions).map(lambda x: ((x[0], x[1]), x[-1]))

    alpha = 0.01
    combined_predictions_test_rdd = prediction_rdd_item_based_test.join(prediction_rdd_model_based_test) \
        .mapValues(lambda x: alpha*x[0]+(1-alpha)*x[1])
    combined_predictions_test = combined_predictions_test_rdd.collect()

    with open(output_file_name, 'w') as file:
        file.write('user_id, business_id, prediction\n')
        for i, prediction in enumerate(combined_predictions_test):
            if i == len(combined_predictions_test)-1:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}')
            else:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}\n')