import sys
from pyspark import SparkContext, StorageLevel
from math import sqrt
from time import perf_counter
from itertools import product

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
                print(len(weights))
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

if __name__ == "__main__":
    start = perf_counter()

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_1')
    sc.setLogLevel('ERROR')

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
    prediction_rdd = weight_rdd \
        .map(lambda x: (x[0], item_based_cf_model.predict(user_item_map_train, x[1], x[0], user_item_map, \
            item_user_map, average_user_ratings, average_item_ratings, overall_average_rating)))
    predictions = prediction_rdd.collect()

    ground_truth = user_item_rdd_test.map(lambda x: x[1]).collect()
    n = len(ground_truth)
    test_rmse = sqrt(user_item_rdd_test.join(prediction_rdd).map(lambda x: (x[1][0]-x[1][1])**2).sum()/n)
    print(f'RMSE = {test_rmse}')

    with open(output_file_name, 'w') as file:
        file.write('user_id, business_id, prediction\n')
        for i, prediction in enumerate(predictions):
            if i == len(predictions)-1:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}')
            else:
                file.write(f'{prediction[0][0]},{prediction[0][1]},{prediction[1]}\n')

    end = perf_counter()
    print(f'Time elapsed: {end-start}')