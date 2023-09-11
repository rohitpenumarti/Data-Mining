import sys
from pyspark import SparkContext, StorageLevel
from random import randint
from itertools import combinations

def read_dataset(sc, filepath):
    return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

def compute_jaccard(set1, set2):
    if len(set1.union(set2)) == 0:
        return 0
    return len(set1.intersection(set2))/len(set1.union(set2))

def hash_func(a, b, p, m):
    return lambda x: ((a*x+b)%p)%m

def create_hash_functions(num_hash_functions, num_bins):
    p = 4294967311
    return [hash_func(randint(1, 2**32-1), randint(0, 2**32-1), p, num_bins) for _ in range(num_hash_functions)]

def create_bands(list_to_split, b, r):
    return [list_to_split[r*i:r*(i+1)] for i in range(b)]

if __name__ == "__main__":
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')

    input_rdd = read_dataset(sc, input_file_name)

    user_id_with_index_rdd = input_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    user_id_with_index_map = user_id_with_index_rdd.collectAsMap()
    user_length = user_id_with_index_rdd.count()

    business_user_rdd = input_rdd.map(lambda x: (x[1], user_id_with_index_map[x[0]])).groupByKey().mapValues(set)
    business_user_map = business_user_rdd.collectAsMap()

    hash_functions = create_hash_functions(100, user_length)

    signature_rdd = input_rdd.map(lambda x: (x[1], user_id_with_index_map[x[0]])) \
        .map(lambda x: (x[0], [func(x[1]) for func in hash_functions])).groupByKey().mapValues(list) \
            .map(lambda x: (x[0], list(map(list, zip(*x[1]))))).map(lambda x: (x[0], [min(val) for val in x[1]]))

    b = 100
    r = 1
    split_sig_rdd = signature_rdd.mapValues(lambda x: create_bands(x, b, r))
    candidate_pair_rdd = split_sig_rdd.mapValues(lambda x: [(i, tuple(x[i])) for i in range(len(x))]).flatMapValues(lambda x: x).map(lambda x: (x[1], x[0])) \
        .groupByKey().mapValues(list).filter(lambda x: True if len(x[1]) >= 2 else False).flatMap(lambda x: list(combinations(x[1], 2)))

    sim_threshold = 0.5
    actual_similar_sets = candidate_pair_rdd.map(lambda x: (x, compute_jaccard(business_user_map[x[0]], business_user_map[x[1]]))) \
        .filter(lambda x: True if x[1] >= sim_threshold else False).groupByKey().mapValues(set).mapValues(list).map(lambda x: (tuple(sorted(x[0])), x[1][0]))

    sorted_actual_similar_sets = actual_similar_sets.sortBy(lambda x: [x[0][0], x[0][1]]).collect()

    with open(output_file_name, 'w') as file:
        file.write('business_id_1, business_id_2, similarity\n')
        for i, pair in enumerate(sorted_actual_similar_sets):
            if i == len(sorted_actual_similar_sets)-1:
                file.write(f'{pair[0][0]}, {pair[0][1]}, {pair[1]}')
            else:
                file.write(f'{pair[0][0]}, {pair[0][1]}, {pair[1]}\n')