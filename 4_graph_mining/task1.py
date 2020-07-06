import itertools
from pyspark.sql import SparkSession
import sys
import time
from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    start_time = time.time()
    # filter_threshold = sys.argv[1]
    # input_file_path = sys.argv[2]
    # output_file_path = sys.argv[3]
    filter_threshold = '7'
    input_file_path = 'data/ub_sample_data.csv'
    output_file_path = 'task1.txt'
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("WARN")
    ss = SparkSession.builder.config('spark.driver.memory', '4G').config('spark.executor.memory', '4G').getOrCreate()
    input_lines = sc.textFile(input_file_path).map(lambda x : x.split(',')).map(lambda x:(x[0], x[1])).filter(lambda x: x[0]!= "user_id").groupByKey().mapValues(lambda x: list(x))
    ub_dict = input_lines.collectAsMap()
    
    edges = []
    points = set()
    for x in list(itertools.combinations(ub_dict.keys(), 2)):
        if len(set(ub_dict[x[0]]).intersection(set(ub_dict[x[1]]))) >= int(filter_threshold):
            edges.append(x)
            edges.append((x[1],x[0]))
            points.add(x[0])
            points.add(x[1])
    points_df = sc.parallelize(list(points)).map(lambda x:(x,))
    points_df = ss.createDataFrame(points_df, ['id'])
    edges_df = sc.parallelize(edges)
    edges_df = ss.createDataFrame(edges_df, schema=['src', 'dst'])
    graph = GraphFrame(points_df, edges_df)
    lpa_graph = graph.labelPropagation(maxIter=5)
    communities = lpa_graph.rdd.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: sorted(list(x[1]))).sortBy(lambda x: (len(x), x))

    result = communities.collect()
    # output
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.writelines(str(line)[1:-1] + "\n")
        output_file.close()
    print('Duration:', (time.time()-start_time))
