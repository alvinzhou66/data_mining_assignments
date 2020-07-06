import itertools
import json
import math
import sys
import time
from pyspark import SparkConf, SparkContext
def extendList(x,y):
    if x != None and y != None:
        return list(x).extend(y)
    else:
        return x
def listToDict(input_list):
    result = {}
    for item in input_list:
        (key, value), = item.items()
        result[key] = value
    return result
def checkCoRatedNum(dict1, dict2):
    if dict1 != None and dict2 != None:
        if len(set(dict1.keys()).intersection(set(dict2.keys()))) >= 3:
            return True
        else:
            return False
    return False
def pearson(c1, c2):
    co_rated = list(set(c1.keys()).intersection(set(c2.keys())))
    star1 = []
    star2 = []
    for x in co_rated:
        star1.append(c1[x])
        star2.append(c2[x])
    average1 = sum(star1)/len(star1)
    average2 = sum(star2)/len(star2)
    above = 0
    for x in zip(star1, star2):
        num = (x[0]-average1)*(x[1]-average2)
        above += num
    below1 = 0
    below2 = 0
    for x in star1:
        num = (x-average1)**2
        below1 += num
    for x in star2:
        num = (x-average2)**2
        below2 += num
    below = math.sqrt(below1)*math.sqrt(below2)
    if above != 0 and below != 0:
        return above/below
    else:
        return 0
def jaccard(c1, c2):
    if c1 != 6666666 and c2 != 6666666:
        if len(set(c1.keys()).intersection(set(c2.keys()))) >= 3:
            result = len(set(c1.keys()).intersection(set(c2.keys())))/len(set(c1.keys()).union(set(c2.keys())))
            return result
    return 0

if __name__ == '__main__':
    start_time = time.time()
    # input_file_path = 'data/train_review2.json'
    # output_file_path = 'task3item.model'
    # model_type = 'item_based'
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    model_type = sys.argv[3]

    configuration = SparkConf().setMaster("local").setAppName("task3_train").set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext(conf=configuration)

    review = sc.textFile(input_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'], x['stars']))

    user_id = review.map(lambda x: x[0]).distinct().sortBy(lambda x: x).collect()
    user_dict = {}
    for i in range(len(user_id)):
        user_dict[user_id[i]] = i
    user_dict2 = {}
    for i in range(len(user_id)):
        user_dict2[i] = user_id[i]
    
    business_id = review.map(lambda x: x[1]).distinct().sortBy(lambda x: x).collect()
    business_dict = {}
    for i in range(len(business_id)):
        business_dict[business_id[i]] = i
    business_dict2 = {}
    for i in range(len(business_id)):
        business_dict2[i] = business_id[i] 
    
    if model_type == 'item_based':
        prediction_model = []
        bus_user_star = review.map(lambda x: (business_dict[x[1]], (user_dict[x[0]], x[2]))).groupByKey()
        bus_user_star = bus_user_star.filter(lambda x: len(list(x[1])) >= 3).mapValues(lambda x: [{y[0]: y[1]} for y in x]).mapValues(lambda x: listToDict(x))
        
        business_user_table = bus_user_star.map(lambda x: (x[0], x[1])).collect()
        business_user_dict = {}
        for i in range(len(business_user_table)):
            business_user_dict[business_user_table[i][0]] = business_user_table[i][1]
        
        candidate_business = bus_user_star.map(lambda x: x[0]).coalesce(4)
        check_similarity = candidate_business.cartesian(candidate_business).filter(lambda x: checkCoRatedNum(business_user_dict[x[0]], business_user_dict[x[1]]))
        check_similarity = check_similarity.map(lambda x: (x, pearson(business_user_dict[x[0]], business_user_dict[x[1]]))).filter(lambda x: x[1]>0).collect()
        for item in check_similarity:
            line = {}
            line['b1'] = business_dict2[item[0][0]]
            line['b2'] = business_dict2[item[0][1]]
            line['sim'] = item[1]
            prediction_model.append(line)    
    elif model_type == 'user_based':
        def hashfunction(number_of_hashfuncs, number_of_bins, x):
            result = []
            for a,b in zip(range(2,number_of_hashfuncs+2), range(10, number_of_hashfuncs+130, 3)):
                result.append((a*x+b)%number_of_bins)
            return result
        prediction_model = []
        bus_user_star = review.map(lambda x: (business_dict[x[1]], (user_dict[x[0]], x[2]))).groupByKey().filter(lambda x: len(list(x[1])) >= 3)
        
        num_buckets = 2*len(bus_user_star.collect())
        user_id_hushed = bus_user_star.flatMap(lambda x: [(y[0], hashfunction(60, num_buckets, x[0])) for y in x[1]])
        def findMinOverLists(l1, l2):
            minList = []
            for i in range(len(l1)):
                minList.append(min(l1[i], l2[i]))
            return minList
        signiture_matrix = user_id_hushed.reduceByKey(findMinOverLists) 
        def listToChunk(input_list, number_of_chunks):
            result = []
            number_of_rows_inAChunk = len(input_list)//number_of_chunks
            for i in range(0, len(input_list), number_of_rows_inAChunk):
                result.append((int((i+number_of_rows_inAChunk)/number_of_rows_inAChunk), tuple(input_list[i:i+number_of_rows_inAChunk])))
            return result 
        split_matrix = signiture_matrix.flatMap(lambda x: [(tuple(band), x[0]) for band in listToChunk(x[1], 60)])
        def createPairs(input_list):
            return list(itertools.combinations(input_list, 2))
        candidate_users = split_matrix.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>=2).flatMap(lambda x: createPairs(x)).distinct()
        
        user_business_table = bus_user_star.flatMap(lambda x: [(y[0], (x[0], y[1])) for y in x[1]]).groupByKey().mapValues(lambda x: list(set(x))).filter(lambda x: len(x[1]) >= 3)
        user_business_table = user_business_table.mapValues(lambda x: [{y[0]: y[1]} for y in x]).mapValues(lambda x: listToDict(x)).collect()
        user_business_dict = {}
        for i in range(len(user_business_table)):
            user_business_dict[user_business_table[i][0]] = user_business_table[i][1]
        
        check_similarity = candidate_users.filter(lambda x: jaccard(user_business_dict.get(x[0], 6666666), user_business_dict.get(x[1], 6666666)) >= 0.01)
        check_similarity = check_similarity.map(lambda x: (x, pearson(user_business_dict[x[0]], user_business_dict[x[1]]))).filter(lambda x: x[1]>0).collect()
        for item in check_similarity:
            line = {}
            line['u1'] = user_dict2[item[0][0]]
            line['u2'] = user_dict2[item[0][1]]
            line['sim'] = item[1]
            prediction_model.append(line)    
    result = prediction_model
    # output
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.write(str(json.dumps(line))+'\n')
        output_file.close()
    print('Duration:', (time.time()-start_time))