import csv
import json
import os
import sys
import time
from math import sqrt
from operator import add
from classes import *
from pyspark import SparkConf, SparkContext   

if __name__ == '__main__':
    start_time = time.time()
    # define input variables
    input_folder_path = "data/test4"
    num_of_cluster = int("8")
    output_file_path = "cluster4.json"
    intermediate_file_path = "intermediate4.csv"
    # input_folder_path = sys.argv[1]
    # num_of_cluster = int(sys.argv[2])
    # output_file_path = sys.argv[3]
    # intermediate_file_path = sys.argv[4]
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("WARN")

    # first Kmeans round
    input_lines = sc.textFile(input_folder_path + "/" + os.listdir(input_folder_path)[0]).map(lambda x: x.split(",")).map(lambda x: (int(x[0]), [float(y) for y in x[1:]]))
    point_count = input_lines.count()
    sample_size = int(point_count*0.02)
    print('sample size:',sample_size)
    sample1 = input_lines.filter(lambda x: x[0]<sample_size)
    sample = sample1.collect()
    input_lines = input_lines.subtractByKey(sample1)
    DS = {}
    CS = {}
    RS = {}
    initialKmeans = KMeans(sample, k = num_of_cluster, current_cluster_id=0, RS=1)
    if -1 in initialKmeans:
        for point in initialKmeans[-1].getInput():
            RS[point[0]] = point
    for index, c in initialKmeans.items():
        if index != -1:
            DS[index] = CreateSet(c)
    if len(RS) > num_of_cluster*4:
        rsKmeans = KMeans(RS.values(), num_of_cluster*2, current_cluster_id=num_of_cluster, RS=1)
        RS = {}
        for index, c in rsKmeans.items():
            if index == -1:
                for point in c.getInput():
                    RS[point[0]] = point
            if c.points_count == 1:
                RS[c.getInput()[0][0]] = c.getInput()[0]
            elif c.points_count != 1:
                CS[index] = CreateSet(c)
    # BFR process
    header = ['round_id','nof_cluster_discard','nof_point_discard','nof_cluster_compression','nof_point_compression','nof_point_retained']
    with open(intermediate_file_path, 'w+', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
    for i in range(len(os.listdir(input_folder_path))):
        print('currently working on',i,"file")
        if i != 0:
            input_lines = sc.textFile(input_folder_path + "/" + os.listdir(input_folder_path)[i]).map(lambda x: x.split(",")).map(lambda x: ([x[0], [float(y) for y in x[1:]]])).persist()
            point_count += input_lines.count()
        # DS
        select_ds = input_lines.map(lambda x: toPoint(x)).map(lambda x: x.selectDS(DS, 2)).persist()
        selected_ds = select_ds.filter(lambda x: x[0] != -1).map(lambda x: (x[0],[x[1]])).reduceByKey(add).map(rddSUM).collect()
        for point in selected_ds: # point: (x[0], rddsum, rddsumq, points)
            DS[point[0]] = DS[point[0]].updateSUM(point[3], point[1], point[2])      
        # CS
        select_cs = select_ds.filter(lambda x: x[0] == -1).map(lambda x: x[1].selectDS(CS, 2)).persist()
        selected_cs = select_cs.filter(lambda x: x[0] != -1).map(lambda x: (x[0],[x[1]])).reduceByKey(add).map(rddSUM).collect()
        for point in selected_cs:
            CS[point[0]] = CS[point[0]].updateSUM(point[3], point[1], point[2])
        # RS
        select_rs = select_cs.filter(lambda x: x[0] == -1).map(lambda x: [x[1].index]+[x[1].vector]).collect()
        for point in select_rs:
            RS[point[0]] = point
        # collect RS to new CS
        if CS:
            current_cluster = max(list(CS.keys()))+1
        else:
            current_cluster = 0
        if len(RS) > num_of_cluster*4:
            rsKmeans = KMeans(RS.values(), num_of_cluster*2, current_cluster_id=current_cluster, RS=1)        
            RS = {}
            for index, c in rsKmeans.items():
                if index == -1:
                    for point in c.getInput():
                        RS[point[0]] = point
                if c.points_count == 1:
                    RS[c.getInput()[0][0]] = c.getInput()[0]
                elif c.points_count != 1:
                    CS[index] = CreateSet(c)
        # combine CS
        if CS:
            cs_distance = []
            for index, cs in CS.items():
                cs_distance.append(cs.combine(CS))
            combined_item = min(cs_distance, key=lambda x: x[2])
            if combined_item[2] != 999999999999:
                CS[combined_item[0]] = CS[combined_item[0]].addFromCS(CS[combined_item[1]])
                del CS[combined_item[1]]
            else:
                break
        # combine CS with DS
        combined_cs = []
        for index, cs in CS.items():
            combined_item = cs.combine(DS)
            if combined_item[2] != 999999999999:
                DS[combined_item[1]] = DS[combined_item[1]].addFromCS(cs)
                combined_cs.append(index)
        for index in combined_cs:
            del CS[index]
        # prepare output parameters
        # nof_point_discard0 = 0
        # for point in DS.values():
        #     nof_point_discard0 += point.N
        nof_point_compression = 0
        for point in CS.values():
            nof_point_compression += point.N
        nof_point_retained = len(RS)
        nof_cluster_compression = len(CS)
        nof_point_discard = point_count-nof_point_compression-nof_point_retained
        with open(intermediate_file_path, 'a') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([i+1, num_of_cluster, nof_point_discard, nof_cluster_compression, nof_point_compression, nof_point_retained])        
 
    # output
    result = {}
    for index, ds in DS.items():
        for p in ds.points:
            result[str(p)] = index
    for index, cs in CS.items():
        for p in cs.points:
            result[str(p)] = -1
    for p in RS.values():
        result[str(p[0])] = -1
    with open(output_file_path, 'w+') as output_file:
        json.dump(result, output_file)
    print('Duration:', (time.time()-start_time))