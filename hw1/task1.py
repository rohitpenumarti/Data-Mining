from pyspark import SparkContext, StorageLevel
import json
import sys

if __name__=='__main__':
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')

    rdd = sc.textFile(review_filepath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK) \

    n_review = rdd.count()
    n_review_2018 = rdd.filter(lambda x: True if '2018' in x['date'] else False).count()
    n_user = rdd.map(lambda x: x['user_id']).distinct().count()
    top10_user = rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a, b: a+b).sortBy(lambda x: [-x[1], x[0]]).take(10)
    n_business = rdd.map(lambda x: x['business_id']).distinct().count()
    top10_business = rdd.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a+b).sortBy(lambda x: [-x[1], x[0]]).take(10)

    top10_user = [list(tup) for tup in top10_user]
    top10_business = [list(tup) for tup in top10_business]

    res = {}
    res['n_review'] = n_review
    res['n_review_2018'] = n_review_2018
    res['n_user'] = n_user
    res['top10_user'] = top10_user
    res['n_business'] = n_business
    res['top10_business'] = top10_business

    with open(output_filepath, 'w') as file:
        file.write(json.dumps(res))
