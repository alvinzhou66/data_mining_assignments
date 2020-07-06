#%%
import sys
import time
from itertools import combinations, islice
from operator import add
from pyspark import SparkContext, SparkConf

def generateKItemsets(L2, K):
    candidates = []
    for i in range(len(L2)-1):
        for j in range(i+1, len(L2)):
            if(L2[i][:K-2] == L2[j][:K-2]):
                # c = list(set(L2[i])).append(L2[j][K-2])
                # c.sort()
                c = list(set(L2[i])|set(L2[j]))
                c.sort()
                if c not in candidates:
                    candidates.append(c)
    return candidates
def apriori(line_list, itemsets, support):       
    # K=1
    basket_a = list(line_list)
    L=[]
    k=1   
    for i in itemsets:
        count = 0
        for item in basket_a:
            if i in item:
                count += 1
        if count >= support:
            L.append(i)
    L.sort()
    len_l1 = len(L)
    result = [[x] for x in L]
    k += 1

    # K=2
    candidates = list()
    
    for x in combinations(L,2):
        candidate = list(x)
        candidate.sort()
        candidates.append(candidate)
    candidates.sort()
    L.clear()
    for c in candidates:
        count = 0
        for i in basket_a:
            if set(c).issubset(i):
                count += 1
        if count >= support:
            L.append(c)
    L.sort()
    result.extend(L)
    k += 1

    # K>2
    while k != len_l1:
        candidates.clear()
        candidates = generateKItemsets(L, k)
        if(len(candidates)==0):
            break
        candidates.sort()
        L.clear()
        for c in candidates:
            count = 0
            for i in basket_a:
                if set(c).issubset(i):
                    count += 1
            if count >= support:
                L.append(c)
        L.sort()
        result.extend(L)
        k += 1
    return result
def count_frequent_itemset(line_list, candidates):
    basket_c = list(line_list)
    result = []
    for c in candidates:
        count = 0
        for i in basket_c:
            if set(c).issubset(i):
                count += 1
        result.append([c, count])
    return result
def output_format(output_result):
    m = 1
    string = ""
    for result in output_result:
        if len(result)==1:
            string += (str(result)[:-2]+'),')
        elif len(result) != m:
            string = string[:-1]
            string += '\n\n'
            m = len(result)
            string += (str(result)+',')
        else:
            string += (str(result)+',')
    return string[:-1]

if __name__ == '__main__':
    start_time = time.time()
    # case_number = "1"
    # support = "4"
    # input_file_path = "data/small2.csv"
    # output_file_path = "task1_small1.txt"

    case_number = sys.argv[1] 
    support = sys.argv[2]
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)

    input_file = sc.textFile(input_file_path)
    small_file = input_file.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
    if int(case_number) == 1:
        basket = small_file.map(lambda x: (x.split(',')[0], x.split(',')[1])).groupByKey().mapValues(lambda x: sorted(list(set(list(x))))).map(lambda x: x[1])

    elif int(case_number) == 2:
        basket = small_file.map(lambda x: (x.split(',')[1], x.split(',')[0])).groupByKey().mapValues(lambda x: sorted(list(set(list(x))))).map(lambda x: x[1])
    sup_part = int(support)/input_file.getNumPartitions()
    total_item= list(set(basket.flatMap(lambda x: x).collect()))
    son_pass1 = basket.mapPartitions(lambda x: apriori(x, total_item, sup_part)).map(lambda x: tuple(x)).distinct().sortBy(lambda x: (len(x), x))
    son_pass1_collect = son_pass1.collect()
    son_pass2 = basket.mapPartitions(lambda x: count_frequent_itemset(x, son_pass1_collect)).reduceByKey(add).filter(lambda x: x[1]>=int(support)).map(lambda x: x[0]).sortBy(lambda x: (len(x), x))
    son_pass2_collect = son_pass2.collect()
    with open(output_file_path, 'w+') as output_file:
        output_str = 'Candidates:\n' + output_format(son_pass1_collect) + '\n\n' + 'Frequent Itemsets:\n' + output_format(son_pass2_collect)
        output_file.write(output_str)
        output_file.close()
    print('Duration:', (time.time()-start_time))
# %%
