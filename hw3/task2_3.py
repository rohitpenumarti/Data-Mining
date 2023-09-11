import sys
import json
from pyspark import SparkContext, StorageLevel
from time import perf_counter
from xgboost import XGBRegressor
from math import sqrt, isnan
from itertools import product
from pandas import DataFrame
from datetime import date, datetime

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

    def __init__(self, filename, review_json_filename, business_json_filename, user_json_filename):
        self.filename = filename
        self.review_json_filename = review_json_filename
        self.business_json_filename = business_json_filename
        self.user_json_filename = user_json_filename

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
        user_rdd = self.read_json_dataset(sc, self.user_json_filename)
        today = date(2022, 10, 28)

        filtered_user_rdd = user_rdd.filter(lambda x: True if x['user_id'] in rdd else False).persist(StorageLevel.MEMORY_AND_DISK)

        rounded_business_rating_rdd = business_rdd.map(lambda x: (x['business_id'], x['stars']))
        business_latitude_rdd = business_rdd.map(lambda x: (x['business_id'], x['latitude']))
        business_longitude_rdd = business_rdd.map(lambda x: (x['business_id'], x['longitude']))
        business_metrics_rdd = rounded_business_rating_rdd.join(business_latitude_rdd).join(business_longitude_rdd).mapValues(lambda x: self._flatten_tuple(x))
        
        average_user_review_rdd = filtered_user_rdd.map(lambda x: (x['user_id'], x['average_stars']))
        num_reviews_by_user_rdd = filtered_user_rdd.map(lambda x: (x['user_id'], x['review_count']))
        avg_review_compliments_rdd = filtered_user_rdd.map(lambda x: (x['user_id'], sum([x['compliment_cool'], x['compliment_funny']])/x['review_count']) \
            if x['review_count'] > 0 else (x['user_id'], 0))
        yelp_user_duration_rdd = filtered_user_rdd.map(lambda x: (x['user_id'], x['yelping_since'])) \
            .mapValues(lambda x: (today-datetime.strptime(x, '%Y-%m-%d').date()).days)
        min_user_duration = yelp_user_duration_rdd.map(lambda x: x[1]).min()
        yelp_user_duration_rdd = yelp_user_duration_rdd.mapValues(lambda x: x-min_user_duration+1)
        reviews_per_year_rdd = num_reviews_by_user_rdd.join(yelp_user_duration_rdd).mapValues(lambda x: (x[0]*365)/x[1])

        user_metrics_rdd = average_user_review_rdd.join(num_reviews_by_user_rdd).join(avg_review_compliments_rdd) \
            .mapValues(lambda x: self._flatten_tuple(x)).join(reviews_per_year_rdd).mapValues(lambda x: self._flatten_tuple(x))
        return business_metrics_rdd, user_metrics_rdd

    def generate_dataset(self, sc, business_metrics_rdd, user_metrics_rdd, is_test=False):
        rdd = self.read_csv_dataset(sc, self.filename)
        today = date(2022, 10, 28)
        
        business_metrics_map = business_metrics_rdd.collectAsMap()
        user_metrics_map = user_metrics_rdd.collectAsMap()
        business_rdd = self.read_json_dataset(sc, self.business_json_filename)
        user_rdd = self.read_json_dataset(sc, self.user_json_filename)

        num_unique_businesses = business_rdd.count()
        average_rounded_business_rating = round(2*business_rdd.map(lambda x: x['stars']) \
            .filter(lambda x: True if isnan(float(x)) else False).sum()/num_unique_businesses)/2
        average_latitude = business_rdd.map(lambda x: x['latitude']).filter(lambda x: True if x else False).sum()/num_unique_businesses
        average_longitude = business_rdd.map(lambda x: x['longitude']).filter(lambda x: True if x else False).sum()/num_unique_businesses

        num_unique_users = user_rdd.count()
        average_user_rating = user_rdd.map(lambda x: x['average_stars']).filter(lambda x: True if isnan(float(x)) else False).sum()/num_unique_users

        num_reviews_by_user_rdd = user_rdd.map(lambda x: (x['user_id'], x['review_count']))
        average_num_user_reviews = num_reviews_by_user_rdd.map(lambda x: x[1]).sum()/num_unique_users

        average_num_compliments = user_rdd.map(lambda x: sum([x['compliment_cool'], x['compliment_funny']])/x['review_count'] \
            if x['review_count'] > 0 else 0).sum()/num_unique_users

        yelp_user_duration_rdd = user_rdd.map(lambda x: (x['user_id'], x['yelping_since'])) \
            .mapValues(lambda x: (today-datetime.strptime(x, '%Y-%m-%d').date()).days)
        min_user_duration = yelp_user_duration_rdd.map(lambda x: x[1]).min()
        yelp_user_duration_rdd = yelp_user_duration_rdd.mapValues(lambda x: x-min_user_duration+1)
        average_num_reviews_per_year = num_reviews_by_user_rdd.join(yelp_user_duration_rdd).map(lambda x: (x[1][0]*365)/x[1][1]).sum()/num_unique_users

        if is_test:
            empty_business_test_vals = rdd.map(lambda x: (x[1], x[0])).leftOuterJoin(business_metrics_rdd) \
                .filter(lambda x: True if x[0] not in business_metrics_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_rounded_business_rating=average_rounded_business_rating, \
                        average_latitude=average_latitude, average_longitude=average_longitude, user=False, is_test=is_test)))
            business_metrics_combined_rdd = rdd.map(lambda x: (x[1], x[0])).join(business_metrics_rdd) \
                .map(lambda x: (x[1][0], self._flatten_tuple(x[1], x[0], x[1][0], is_test=is_test))).union(empty_business_test_vals) \
                    .map(lambda x: ((x[0], x[1][0]), (x[1][1], x[1][2], x[1][3])))
            
            empty_user_test_vals = rdd.map(lambda x: (x[0], x[1])).leftOuterJoin(user_metrics_rdd) \
                .filter(lambda x: True if x[0] not in user_metrics_map else False) \
                    .map(lambda x: (x[1][0], self._populate_empty_features(x, average_user_rating=average_user_rating, \
                        average_num_user_reviews=average_num_user_reviews, average_num_compliments=average_num_compliments, \
                            average_num_reviews_per_year=average_num_reviews_per_year, biz=False, is_test=is_test)))
            user_metrics_combined_rdd = rdd.map(lambda x: (x[0], x[1])).join(user_metrics_rdd) \
                .map(lambda x: (x[1][0], self._flatten_tuple(x[1], x[0], x[1][0], is_test=is_test))).union(empty_user_test_vals) \
                    .map(lambda x: ((x[1][0], x[0]), (x[1][1], x[1][2], x[1][3], x[1][4])))
        else:
            empty_business_test_vals = rdd.map(lambda x: (x[1], (x[0], float(x[2])))).leftOuterJoin(business_metrics_rdd) \
                .filter(lambda x: True if x[0] not in business_metrics_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_rounded_business_rating=average_rounded_business_rating, \
                        average_latitude=average_latitude, average_longitude=average_longitude, user=False)))
            business_metrics_combined_rdd = rdd.map(lambda x: (x[1], (x[0], float(x[2])))).join(business_metrics_rdd) \
                .map(lambda x: (x[1][0][0], self._flatten_tuple(x[1], x[0], x[1][0][0]))).union(empty_business_test_vals) \
                    .map(lambda x: ((x[0], x[1][0]), (x[1][1], x[1][2], x[1][3], x[1][4])))

            empty_user_test_vals = rdd.map(lambda x: (x[0], (x[1], float(x[2])))).leftOuterJoin(user_metrics_rdd) \
                .filter(lambda x: True if x[0] not in user_metrics_map else False) \
                    .map(lambda x: (x[1][0][0], self._populate_empty_features(x, average_user_rating=average_user_rating, \
                        average_num_user_reviews=average_num_user_reviews, average_num_compliments=average_num_compliments, \
                            average_num_reviews_per_year=average_num_reviews_per_year, biz=False)))
            user_metrics_combined_rdd = rdd.map(lambda x: (x[0], (x[1], float(x[2])))).join(user_metrics_rdd) \
                .map(lambda x: (x[1][0][0], self._flatten_tuple(x[1], x[0], x[1][0][0], is_test=is_test))).union(empty_user_test_vals) \
                    .map(lambda x: ((x[1][0], x[0]), (x[1][2], x[1][3], x[1][4], x[1][5])))
        
        if is_test:
            combined_rdd = business_metrics_combined_rdd.join(user_metrics_combined_rdd).map(lambda x: list(self._flatten_tuple(x[1], x[0])))
            column_names = ['user_id', 'business_id', 'rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', \
                'avg_user_compliments', 'reviews_per_year']
        else:
            combined_rdd = business_metrics_combined_rdd.join(user_metrics_combined_rdd).map(lambda x: list(self._flatten_tuple(x[1], x[0])))
            column_names = ['user_id', 'business_id', 'rating', 'rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', \
                'avg_user_compliments', 'reviews_per_year']
        
        feature_matrix_as_list = combined_rdd.collect()
        return DataFrame(feature_matrix_as_list, columns=column_names)

def compute_RMSE(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for t, p in zip(y_true, y_pred):
        sum += (t-p)**2

    return sqrt(sum/n)

if __name__ == "__main__":
    start = perf_counter()
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_3')
    sc.setLogLevel('ERROR')

    train_file_name = folder_path + 'yelp_train.csv'

    ##### ITEM-BASED-CF MODEL #####

    item_based_cf_model = Item_Based_CF(5, 4)

    train_rdd = item_based_cf_model.read_dataset(sc, train_file_name)
    test_rdd = item_based_cf_model.read_dataset(sc, test_file_name)

    average_user_ratings = item_based_cf_model.get_user_averages(train_rdd)
    average_item_ratings = item_based_cf_model.get_item_averages(train_rdd)

    num_user_ratings = item_based_cf_model.get_user_review_count(train_rdd)
    num_item_ratings = item_based_cf_model.get_item_review_count(train_rdd)

    user_item_rdd_train = item_based_cf_model.create_user_item_matrix(train_rdd)
    user_item_map_train = user_item_rdd_train.collectAsMap()
    user_item_rdd_test = item_based_cf_model.create_user_item_matrix(test_rdd)

    item_user_map = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    user_item_map = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    test_map = user_item_rdd_test.collectAsMap()
    overall_average_rating = train_rdd.map(lambda x: float(x[2])).sum()/train_rdd.count()

    weight_rdd = item_based_cf_model.compute_weight(test_rdd, user_item_map_train, user_item_map, item_user_map)
    prediction_rdd_item_based = weight_rdd \
        .map(lambda x: (x[0], item_based_cf_model.predict(user_item_map_train, x[1], x[0], user_item_map, \
            item_user_map, average_user_ratings, average_item_ratings, overall_average_rating)))

    ground_truth = user_item_rdd_test.map(lambda x: x[1]).collect()
    n = len(ground_truth)
    test_rmse_item_based = sqrt(user_item_rdd_test.join(prediction_rdd_item_based).map(lambda x: (x[1][0]-x[1][1])**2).sum()/n)
    print(f'ITEM-BASED-CF\nTest RMSE: {test_rmse_item_based}')

    ##### XGBOOST MODEL #####

    review_json_filename = folder_path + 'review_train.json'
    business_json_filename = folder_path + 'business.json'
    user_json_filename = folder_path + 'user.json'
    train_file_name = folder_path + 'yelp_train.csv'

    train_dataset = Dataset(train_file_name, review_json_filename, business_json_filename, user_json_filename)
    business_metrics_rdd, user_metrics_rdd = train_dataset.generate_metrics(sc)
    train_feature_matrix = train_dataset.generate_dataset(sc, business_metrics_rdd, user_metrics_rdd)
    print(f'Expected: {train_dataset.read_csv_dataset(sc, train_file_name).count()}, Actual: {len(train_feature_matrix)}')

    test_dataset = Dataset(test_file_name, review_json_filename, business_json_filename, user_json_filename)
    test_feature_matrix = test_dataset.generate_dataset(sc, business_metrics_rdd, user_metrics_rdd, True)
    print(f'Expected: {train_dataset.read_csv_dataset(sc, test_file_name).count()}, Actual: {len(test_feature_matrix)}')
    
    X_train = train_feature_matrix[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', \
        'num_user_reviews', 'avg_user_compliments', 'reviews_per_year']]
    y_train = train_feature_matrix['rating']

    xgb = XGBRegressor().fit(X_train, y_train)

    y_pred_train = list(xgb.predict(X_train))
    train_rmse = compute_RMSE(list(y_train), y_pred_train)

    X_test = test_feature_matrix[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', \
        'num_user_reviews', 'avg_user_compliments', 'reviews_per_year']]
    y_pred_test = list(xgb.predict(X_test))

    test_features_list = test_feature_matrix[['user_id', 'business_id']].values.tolist()
    
    test_set_with_predictions = []
    for i, feats in enumerate(test_features_list):
        test_set_with_predictions.append(feats.copy())
        test_set_with_predictions[-1].append(y_pred_test[i])

    prediction_rdd_model_based = sc.parallelize(test_set_with_predictions).map(lambda x: ((x[0], x[1]), x[-1]))

    test_rdd = test_dataset.read_csv_dataset(sc, test_file_name).map(lambda x: ((x[0], x[1]), float(x[2])))
    test_rmse_model_based = sqrt(prediction_rdd_model_based.join(test_rdd).map(lambda x: (x[1][0]-x[1][1])**2).sum()/test_rdd.count())

    print(f'MODEL-BASED-CF\nTrain RMSE: {train_rmse}; Test RMSE: {test_rmse_model_based}')

    alpha = 0.01
    combined_predictions_rdd = prediction_rdd_item_based.join(prediction_rdd_model_based).mapValues(lambda x: alpha*x[0]+(1-alpha)*x[1])
    combined_predictions = combined_predictions_rdd.collect()

    with open(output_file_name, 'w') as file:
        file.write('user_id, business_id, prediction\n')
        for i, prediction in enumerate(combined_predictions):
            if i == len(combined_predictions)-1:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}')
            else:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}\n')

    prediction_test_rdd = test_dataset.read_csv_dataset(sc, output_file_name).map(lambda x: ((x[0], x[1]), float(x[2])))
    test_rmse_hybrid = sqrt(prediction_test_rdd.join(test_rdd).map(lambda x: (x[1][0]-x[1][1])**2).sum()/test_rdd.count())
    
    print(f'HYBRID-CF\nRMSE: {test_rmse_hybrid}')

    end = perf_counter()
    print(f'Time elapsed: {end-start}')