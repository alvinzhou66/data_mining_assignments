#%%
import sys
import json
from operator import add
from pyspark import SparkContext
if __name__ == '__main__':

    review_file_path = "data/review_sample.json"
    output_file_path = "output3.json"
    partition_type = "default"  # either "default" or "customized"
    n_partition = '3'
    n = '1'
    # review_file_path = sys.argv[1]
    # output_file_path = sys.argv[2]
    # partition_type = sys.argv[3]
    # n_partition = sys.argv[4]
    # n = sys.argv[5]
    sc = SparkContext.getOrCreate()
    review = sc.textFile(review_file_path).map(lambda x: json.loads(x))
    business_count = review.map(lambda x: (x['business_id'], 1))
    result = {}
    if partition_type == "default":
        # reference: https://stackoverflow.com/questions/46032320/apache-spark-get-number-of-records-per-partition
        result['n_partitions'] = business_count.getNumPartitions()
        result['n_items'] = business_count.glom().map(len).collect()
        result['result'] = business_count.reduceByKey(add).filter(lambda x: x[1] > int(n)).collect()
    else:
        # try to improve performance by assigning tasks evenly onto each partitioner
        def assignPartition(key):
            return ord(key[0])%int(n_partition)
        business_count = business_count.partitionBy(int(n_partition), assignPartition)
        result['n_partitions'] = business_count.getNumPartitions()
        result['n_items'] = business_count.glom().map(len).collect()
        result['result'] = business_count.reduceByKey(add).filter(lambda x: x[1] > int(n)).collect()
    print(result)
    with open(output_file_path, 'w+') as output_file:
        json.dump(result, output_file)
    output_file.close()
# %%
