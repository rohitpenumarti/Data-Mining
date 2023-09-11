from pyspark import SparkContext, StorageLevel
import sys
from time import perf_counter
from itertools import combinations, product
from math import ceil

def preprocess_data(rdd, output_file_path):
    backslash_quotes = '\"'
    customer_product_rdd = rdd.map(lambda x: (f"{x[0].strip(backslash_quotes)}-{x[1].strip(backslash_quotes).strip('0')}", int(x[5].strip(backslash_quotes).strip("0"))))
    customer_product = customer_product_rdd.collect()

    with open(output_file_path, 'w') as file:
        file.write('DATE-CUSTOMER_ID,PRODUCT_ID\n')
        for cp in customer_product:
            file.write(f'{cp[0]},{cp[1]}\n')

def filter_data(k, rdd):
    return rdd.groupByKey().mapValues(set).filter(lambda x: True if len(x[1]) > k else False).flatMapValues(lambda x: x)

def read_dataset(sc, filepath):
    rdd = sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)
    return rdd

def write_file(file, itemset_type, frequent_itemsets):
    if itemset_type == 'Candidates':
        file.write(f'{itemset_type}:\n')
    else:
        file.write(f'\n\n{itemset_type}:\n')
    curr_length = 1
    for i, fi in enumerate(frequent_itemsets):
        new_length = len(fi[0])
        if new_length > curr_length:
            file.write('\n\n')
            curr_length = new_length
        
        if i+1 < len(frequent_itemsets):
            if len(fi[0]) < len(frequent_itemsets[i+1][0]):
                if len(fi[0]) == 1:
                    file.write(f'(\'{fi[0][0]}\')')
                else:
                    file.write(f'{fi[0]}')
            else:
                if len(fi[0]) == 1:
                    file.write(f'(\'{fi[0][0]}\'),')
                else:
                    file.write(f'{fi[0]},')
        else:
            if len(fi[0]) == 1:
                file.write(f'(\'{fi[0][0]}\')')
            else:
                file.write(f'{fi[0]}')

def create_baskets(data):
    intermediate_baskets = list(map(lambda x: (x[0], list(x[1:])), \
        sorted(list(map(lambda x: (x[0], x[1]), data)), key=lambda x: [x[0], int(x[1])])))
    baskets = {}
    for b in intermediate_baskets:
        if b[0] not in baskets:
            baskets[b[0]] = set()
        baskets[b[0]].add(b[1][0])
    return baskets

def get_max_length_basket(baskets):
    basket_lengths = list(map(lambda x: len(x[1]), baskets.items()))
    if basket_lengths:
        return max(basket_lengths)
    else:
        return 0

def apriori(s, init):
    ps = ceil(s*len(init))
    baskets = create_baskets(init)

    max_k = get_max_length_basket(baskets)
    candidate_itemsets = []
    frequent_itemsets = []

    candidate_itemsets.append(set(map(lambda x: (x[1],), init)))
    item_counts = dict(map(lambda x: ((x[1],), [r[1] for r in init].count(x[1])), init))
    curr_frequent_itemsets_with_counts = set(filter(lambda x: True if x[1] >= ps else False, item_counts.items()))
    frequent_itemsets.append(set([tup[0] for tup in curr_frequent_itemsets_with_counts]))
    frequent_itemsets_with_counts = list(curr_frequent_itemsets_with_counts)
    for i in range(1, max_k):
        if not frequent_itemsets[i-1]:
            break
        else:
            previous_unique_frequent_items = set(sum(frequent_itemsets[i-1], ()))
            previous_infrequent_itemsets = set([itemset for itemset in candidate_itemsets[i-1]-frequent_itemsets[i-1]])

            new_candidates = set(map(lambda itemset_combo: tuple(sorted(itemset_combo[0] + (itemset_combo[1],))) \
                if itemset_combo[1] not in itemset_combo[0] else (), product(frequent_itemsets[i-1], previous_unique_frequent_items)))
            new_candidates.remove(())

            candidates_copy = new_candidates.copy()
            for itemset in candidates_copy:
                for sub_itemset in combinations(itemset, i):
                    if sub_itemset in previous_infrequent_itemsets:
                        new_candidates.remove(itemset)
                        break

            candidate_itemsets.append(new_candidates)

            itemset_counts = {}
            for itemset in candidate_itemsets[i]:
                count = 0
                for b in baskets:
                    if set(itemset).issubset(baskets[b]):
                        count += 1

                itemset_counts[itemset] = count
            
            curr_frequent_itemsets_with_counts = set(filter(lambda x: True if x[1] >= ps else False, itemset_counts.items()))
            frequent_itemsets.append(set([tup[0] for tup in curr_frequent_itemsets_with_counts]))
            frequent_itemsets_with_counts.extend(list(curr_frequent_itemsets_with_counts))
    return frequent_itemsets_with_counts

def son(s, init):
    total_count = init.count()
    num_partitions = init.getNumPartitions()
    candidate_frequent_itemsets = init.partitionBy(num_partitions, lambda x: hash(x)%num_partitions) \
        .mapPartitions(lambda x: apriori(s/total_count, list(x))).reduceByKey(lambda a, b: a+b).sortBy(lambda x: [len(x[0]), x[0]])
    truly_frequent_itemsets = candidate_frequent_itemsets.filter(lambda x: True if x[1] >= s else False)
    return candidate_frequent_itemsets.collect(), truly_frequent_itemsets.collect()

if __name__ == "__main__":
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('ERROR')

    start = perf_counter()

    rdd = read_dataset(sc, input_file_path)
    preprocess_data(rdd, 'task2_preprocessed_data.csv')
    customer_product_rdd = read_dataset(sc, 'task2_preprocessed_data.csv')
    filtered_rdd = filter_data(filter_threshold, customer_product_rdd)
    
    candidate_frequent_itemsets, truly_frequent_itemsets = son(support, filtered_rdd)

    with open(output_file_path, 'w') as file:
        write_file(file, 'Candidates', candidate_frequent_itemsets)
        write_file(file, 'Frequent Itemsets', truly_frequent_itemsets)

    end = perf_counter()
    print(f'Duration: {end-start}')