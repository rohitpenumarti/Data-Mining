import sys
import json
from pyspark import SparkContext, StorageLevel
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from math import isnan
from pandas import DataFrame
from ast import literal_eval
import pandas as pd
from collections import defaultdict
from datetime import date, datetime
import pickle

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
    ##### folder_path: folder path of json datasets. Same parameter as in competition.py #####
    ##### output_file_name: file name for the outputted pickle file #####
    folder_path = sys.argv[1]
    output_file_name = sys.argv[2]

    sc = SparkContext('local[*]', 'train')
    sc.setLogLevel('ERROR')

    train_file_name = folder_path + 'yelp_train.csv'
    review_json_filename = folder_path + 'review_train.json'
    business_json_filename = folder_path + 'business.json'
    user_json_filename = folder_path + 'user.json'
    tip_json_filename = folder_path + 'tip.json'

    train_dataset = Dataset(train_file_name, review_json_filename, business_json_filename, user_json_filename, tip_json_filename)

    train_rdd = train_dataset.read_csv_dataset(sc, train_file_name)
    user_item_map = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    business_metrics_rdd, user_metrics_rdd = train_dataset.generate_metrics(sc)
    train_feature_matrix = train_dataset.generate_dataset(sc, user_item_map, business_metrics_rdd, user_metrics_rdd)

    X_train = train_feature_matrix[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', \
        'avg_user_compliments', 'reviews_per_year', 'review_count', 'price_range', 'bike_parking', 'good_for_kids', 'has_TV', \
            'outdoor_seating', 'restaurants_delivery', 'restaurants_good_for_groups', 'restaurants_reservations', \
                'restaurants_take_out', 'by_appointment_only', 'wheelchair_accessible', 'open_24_hours', 'drive_thru', \
                    'coat_check', 'restaurants_table_service', 'business_accepts_bitcoin', 'garage', 'lot', 'lunch', 'casual', \
                        'city_state_le', 'name_le', 'total_user_likes', 'avg_user_likes', 'max_business_likes', \
                            'total_business_likes', 'avg_business_likes', 'avg_ub_pair_likes']]
    y_train = train_feature_matrix['rating']

    xgb = XGBRegressor(objective='reg:linear', base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, \
        gpu_id=1, interaction_constraints='', max_delta_step=0, random_state=0, \
            n_jobs=8, num_parallel_tree=1, scale_pos_weight=1, subsample=1, \
                tree_method='exact', validate_parameters=1, verbosity=None, n_estimators=574, \
                    max_depth=17, reg_lambda=34.99906842816348, \
                        gamma=4.886927896270191, learning_rate=0.04262522175688959, \
                            reg_alpha=63, min_child_weight=8, \
                                colsample_bytree=0.7670554549316846).fit(X_train, y_train)

    pickle.dump(xgb, open(output_file_name, "wb"))