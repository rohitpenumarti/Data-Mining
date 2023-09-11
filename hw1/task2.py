from pyspark import SparkContext, StorageLevel
import json
from time import perf_counter
import sys
    

if __name__=='__main__':
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition = int(sys.argv[3])

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('ERROR')

    start_default = perf_counter()
    rdd_default = sc.textFile(review_filepath).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).persist(StorageLevel.MEMORY_AND_DISK)
    n_partition_default = rdd_default.getNumPartitions()
    n_items_default = rdd_default.glom().map(len).collect()
    top10_business_default = rdd_default.reduceByKey(lambda a, b: a+b).sortBy(lambda x: [-x[1], x[0]]).take(10)
    end_default = perf_counter()
    exe_time_default = end_default-start_default

    start_custom = perf_counter()
    rdd_custom = sc.textFile(review_filepath).map(lambda x: json.loads(x)) \
                    .map(lambda x: (x['business_id'], 1)).partitionBy(n_partition, lambda x: hash(x)).persist(StorageLevel.MEMORY_AND_DISK)
    n_partition_custom = rdd_custom.getNumPartitions()
    n_items_custom = rdd_custom.glom().map(len).collect()
    top10_business_custom = rdd_custom.reduceByKey(lambda a, b: a+b).sortBy(lambda x:[-x[1], x[0]]).take(10)
    end_custom = perf_counter()
    exe_time_custom = end_custom-start_custom
    
    res = {}
    res['default'] = {"n_partition": n_partition_default, "n_items": n_items_default, "exe_time": exe_time_default}
    res['customized'] = {"n_partition": n_partition, "n_items": n_items_custom, "exe_time": exe_time_custom}

    with open(output_filepath, 'w') as file:
        file.write(json.dumps(res))