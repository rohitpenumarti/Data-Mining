import sys
from pyspark import SparkContext, StorageLevel
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SQLContext
from graphframes import GraphFrame

def read_dataset(sc, filepath):
    return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

def create_user_business_matrix(rdd):
    return rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)

def generate_graph(user_business_matrix, filter_threshold):
    graph_rdd = user_business_matrix.cartesian(user_business_matrix).filter(lambda x: True if x[0][0] != x[1][0] else False) \
        .map(lambda x: ((x[0][0], x[1][0]), x[0][1].intersection(x[1][1]))) \
            .filter(lambda x: True if len(x[1]) >= filter_threshold else False)
    return graph_rdd

if __name__ == "__main__":
    filter_threshold = int(sys.argv[1])
    input_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')
    sqlContext = SQLContext(sc)

    rdd = read_dataset(sc, input_file_name)
    user_business_matrix = create_user_business_matrix(rdd)
    graph_rdd = generate_graph(user_business_matrix, filter_threshold)
    
    labels = ['src', 'dst']
    schema_v = StructType([StructField('id', StringType(), True)])
    schema_e = StructType([StructField(l, StringType(), True) for l in labels])
    vertices = sqlContext.createDataFrame(graph_rdd.map(lambda x: x[0][0]).distinct().map(lambda x: [x]), schema_v)
    edges = sqlContext.createDataFrame(graph_rdd.map(lambda x: list(x[0])), schema_e)
    
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    grouped_rdd = result.rdd.map(tuple).map(lambda x: (x[1], x[0])).groupByKey().mapValues(sorted)
    sorted_result = grouped_rdd.map(lambda x: x[1]).sortBy(lambda x: [len(x), x[0]]).collect()

    with open(output_file_name, 'w') as file:
        for i, community in enumerate(sorted_result):
            for j, user in enumerate(community):
                if j == len(community)-1:
                    file.write(f'\'{user}\'\n')
                else:
                    file.write(f'\'{user}\', ')