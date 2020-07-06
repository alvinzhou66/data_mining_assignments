import binascii
import sys
import datetime
import json
import csv
from random import sample
from statistics import median_high, mean
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
def countLast0(input_bin):
    result = 0
    for bit in str(input_bin)[2:][::-1]:
        if bit == '0':
            result += 1
        else:
            return result
    return result
def hashfunction(para_pair, number_of_bins, x):
    result = []
    for p in para_pair:
        hash_value = (p[0]*x+p[1])%number_of_bins
        result.append([countLast0(bin(hash_value))])
    return result
def listSum(list1, list2):
    result = []
    for i in range(min(len(list1), len(list2))):
        result.append(list1[i] + list2[i])
    return result
def FM(rdd):    
    timestamp = str(datetime.datetime.now())[:-7]
    estimates = []
    ground_truth = rdd.distinct().count()
    for i in range(0,7):
        zore_count = rdd.map(lambda x: hashfunction(para_pair[6*i:6*i+6], number_of_bins, x)).reduce(lambda x,y: listSum(x,y))
        temp = []
        for l in zore_count:
            temp.append(max(l))
        R = int(2**median_high(temp))
        estimates.append(R)
    estimate_count = mean(estimates)

    with open(output_file_path, 'a+', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([timestamp,ground_truth,estimate_count])
if __name__ == '__main__':
    # port = int(sys.argv[1])
    # output_file_path = sys.argv[2]
    port = 9999
    output_file_path = 'task2.csv'
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel(logLevel='OFF')
    ssc = StreamingContext(sc, 5)
    with open(output_file_path, 'w+', newline = '') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Time','Ground Truth','Estimation'])
        
    input_stream = ssc.socketTextStream('localhost', port).map(lambda x: json.loads(x)).map(lambda x: x['city']).filter(lambda x: x != '').map(lambda x: int(binascii.hexlify(x.encode('utf8')),16))
    # parameters for hash
    number_of_hashfuncs = 42
    number_of_bins = 179
    para_pair = list(zip(sample(range(1, 999), number_of_hashfuncs), sample(range(0, 999), number_of_hashfuncs)))

    slice_window = input_stream.window(30,10).foreachRDD(FM)

    ssc.start()
    ssc.awaitTermination()