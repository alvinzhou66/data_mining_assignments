import binascii
import sys
import time
import json
import csv
from random import sample
from pyspark import SparkConf, SparkContext
def hashfunction(para_pair, number_of_bins, x):
    result = []
    for p in para_pair:
        result.append((p[0]*x+p[1])%number_of_bins)
    return result
def predict(input_line, hashed_cities):
    if input_line != '' and input_line is not None:
        utf_input = int(binascii.hexlify(input_line.encode('utf8')),16)
        hashed_value = hashfunction(para_pair, number_of_bins, utf_input)
        if set(hashed_value).issubset(hashed_cities):
            return 1
        else:
            return 0
    else:
        return 0
if __name__ == '__main__':
    start_time = time.time()
    first_file_path = sys.argv[1]
    second_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    # first_file_path = 'data/business_first.json'
    # second_file_path = 'data/business_second.json'
    # output_file_path = 'task1.csv'
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("WARN")
    input_lines = sc.textFile(first_file_path).map(lambda x: json.loads(x)).map(lambda x: x['city']).distinct().filter(lambda x: x != '').map(lambda x: int(binascii.hexlify(x.encode('utf8')),16))
    # hush parameters
    number_of_hashfuncs = 7
    number_of_bins = 7000
    para_pair = list(zip(sample(range(1, 99), number_of_hashfuncs), sample(range(0, 999), number_of_hashfuncs)))
    
    hashed_cities = input_lines.map(lambda x: hashfunction(para_pair, number_of_bins, x)).reduce(lambda x, y: set(x).union(set(y)))
    predict_result = sc.textFile(second_file_path).map(lambda x: json.loads(x)).map(lambda x: x['city']).map(lambda x: predict(x, hashed_cities)).collect()
    with open(output_file_path, 'w+', newline = '') as output_file:
        writer = csv.writer(output_file, delimiter=' ')
        writer.writerow(predict_result)
    print('Duration:', (time.time()-start_time))