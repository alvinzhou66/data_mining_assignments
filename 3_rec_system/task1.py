#%%
from itertools import combinations
import json
import sys
import time
from pyspark import SparkContext, SparkConf

def hashfunction(number_of_hashfuncs, number_of_bins, x):
    result = []
    for a,b in zip(range(2,number_of_hashfuncs+2), range(10, number_of_hashfuncs+130, 3)):
        result.append((a*x+b)%number_of_bins)
    return result

if __name__ == '__main__':
    start_time = time.time()
    input_file_path = "data/train_review.json"
    output_file_path = "task1.res"
    # input_json_path = sys.argv[1]
    # output_file_path = sys.argv[2]

    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    #extract user and business data
    review = sc.textFile(input_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id']))
    
    user_id = review.map(lambda x: x[0]).distinct().sortBy(lambda x: x).collect()
    user_dict = {}
    for i in range(len(user_id)):
        user_dict[user_id[i]] = i
    
    business_id = review.map(lambda x: x[1]).distinct().sortBy(lambda x: x).collect()
    business_dict = {}
    for i in range(len(business_id)):
        business_dict[business_id[i]] = i
    business_dict2 = {}
    for i in range(len(business_id)):
        business_dict2[i] = business_id[i]    
    #create user & business relational table
    business_user_table = review.map(lambda x: (business_dict[x[1]], user_dict[x[0]])).groupByKey().mapValues(lambda x: list(set(x))).collect()
    business_user_table_dict = {}
    for i in range(len(business_user_table)):
        business_user_table_dict[business_user_table[i][0]] = business_user_table[i][1]
    
    user_business_table = review.map(lambda x: (user_dict[x[0]], business_dict[x[1]])).groupByKey().mapValues(lambda x: list(set(x)))
    # apply hush function
    user_id_hushed = review.map(lambda x: x[0]).distinct().sortBy(lambda x: x).map(lambda x: (user_dict[x], hashfunction(60, 30011, user_dict[x]))) 
    # create signiture matrix
    signiture_matrix = user_business_table.leftOuterJoin(user_id_hushed).map(lambda x: x[1]).flatMap(lambda x: [(index, x[1]) for index in x[0]])
    # apply min-hash
    def findMinOverLists(l1, l2):
        minList = []
        for i in range(len(l1)):
            minList.append(min(l1[i], l2[i]))
        return minList
    signiture_matrix = signiture_matrix.reduceByKey(findMinOverLists)
    # create candidate similar business

    def listToChunk(input_list, number_of_chunks):
        result = []
        number_of_rows_inAChunk = len(input_list)//number_of_chunks
        for i in range(0, len(input_list), number_of_rows_inAChunk):
            result.append((int((i+number_of_rows_inAChunk)/number_of_rows_inAChunk), tuple(input_list[i:i+number_of_rows_inAChunk])))
        return result
    split_matrix = signiture_matrix.flatMap(lambda x: [(tuple(chunk), x[0]) for chunk in listToChunk(x[1], 60)])
    def createPairs(input_list):
        return list(combinations(input_list, 2))
    candidate_business = split_matrix.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>=2).flatMap(lambda x: createPairs(x))
    
    # compute jaccard similarity and find truly similar pairs

    result = []
    distinct_business_pairs = set()
    for business_pairs in candidate_business.collect():
        if business_pairs not in distinct_business_pairs:
            distinct_business_pairs.add(business_pairs)
            business1 = business_user_table_dict[business_pairs[0]]
            business2 = business_user_table_dict[business_pairs[1]]
            similarity = float(len(set(business1).intersection(set(business2)))/len(set(business1).union(set(business2))))
            if similarity >= 0.05:
                result.append({"b1": business_dict2[business_pairs[0]], "b2": business_dict2[business_pairs[1]], "sim": similarity})
    # output
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.write(str(json.dumps(line))+'\n')
        output_file.close()
    print('Duration:', (time.time()-start_time))


# %%
