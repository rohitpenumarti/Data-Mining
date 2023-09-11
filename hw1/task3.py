from pyspark import SparkContext, StorageLevel
import json
from time import perf_counter
import sys

if __name__=='__main__':
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]

    sc = SparkContext('local[*]', 'task3')
    sc.setLogLevel('ERROR')

    ##### Spark Sort Implementation 3a/3bm2 #####
    start_spark = perf_counter()
    review_rdd = sc.textFile(review_filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)
    business_rdd = sc.textFile(business_filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)

    businesses_reviewed_rdd = review_rdd.map(lambda x: (x['business_id'], x['stars']))
    business_cities_rdd = business_rdd.map(lambda x: (x['business_id'], x['city']))

    final_rdd = businesses_reviewed_rdd.join(business_cities_rdd) \
        .map(lambda x: (x[1][1], (x[1][0], 1))).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
            .map(lambda x: (x[0], x[1][0]/x[1][1])).sortBy(lambda x: [-x[1], x[0]]).persist(StorageLevel.MEMORY_AND_DISK)
    top10_cities = final_rdd.take(10)
    end_spark = perf_counter()

    print(top10_cities)
    city_stars = final_rdd.collect()

    with open(output_filepath_question_a, 'w') as file:
        file.write('city,stars\n')
        for cs in city_stars:
            file.write(f'{cs[0]},{cs[1]}\n')

    ##### Python Sort Implementation 3bm1 #####
    start_python = perf_counter()
    review_rdd_python = sc.textFile(review_filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)
    business_rdd_python = sc.textFile(business_filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)

    businesses_reviewed_rdd = review_rdd.map(lambda x: (x['business_id'], x['stars']))
    business_cities_rdd = business_rdd.map(lambda x: (x['business_id'], x['city']))

    final_list = businesses_reviewed_rdd.join(business_cities_rdd) \
        .map(lambda x: (x[1][1], (x[1][0], 1))).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
            .map(lambda x: (x[0], x[1][0]/x[1][1])).collect()

    sorted_top10_cities = sorted(final_list, key=lambda x: [-x[1], x[0]])[:10]
    print(sorted_top10_cities)
    end_python = perf_counter()

    res_3b = {}
    res_3b['m1'] = end_python-start_python
    res_3b['m2'] = end_spark-start_spark
    res_3b['reason'] = '''The sortBy function requires a shuffle which results in a longer runtime in comparison to using the sorted function in python.
                        Since both algorithms have the same time complexity, the shuffle operation makes the spark version slower.'''

    with open(output_filepath_question_b, 'w') as file:
        file.write(json.dumps(res_3b))