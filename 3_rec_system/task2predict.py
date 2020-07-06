import json
import math
import sys
import time
from pyspark import SparkConf, SparkContext

def cosSimilarity(user_p, business_p):
    if user_p != 6666666 and business_p != 6666666:
        user_p = set(user_p)
        business_p = set(business_p)
        # because in our 0/1 matrix, all values are 1, so A*B and sqrt(A)*sqrt(B) have following value
        ab = len(user_p.intersection(business_p))
        sqrt = math.sqrt(len(user_p))*math.sqrt(len(business_p))
        result = ab/sqrt
        return result
    else:
        return 0

if __name__ == '__main__':
    start_time = time.time()

    # input_file_path = "data/test_review.json"
    # model_file_path = "task2.model"
    # output_file_path = "task2.predict"
    # train_file_path = "data/train_review.json"
    input_file_path = sys.argv[1]
    model_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    train_file_path = '$ASNLIB/publicdata/train_review.json'
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration) 

    model = sc.textFile(model_file_path).map(lambda x: json.loads(x))
    review = sc.textFile(train_file_path).map(lambda x: json.loads(x))
    user_id = review.map(lambda x: x['user_id']).distinct().sortBy(lambda x: x).collect()
    user_dict = {}
    for i in range(len(user_id)):
        user_dict[user_id[i]] = i
    user_dict2 = {}
    for i in range(len(user_id)):
        user_dict2[i] = user_id[i]
    business_id = review.map(lambda x: x['business_id']).distinct().sortBy(lambda x: x).collect()
    business_dict = {}
    for i in range(len(business_id)):
        business_dict[business_id[i]] = i
    business_dict2 = {}
    for i in range(len(business_id)):
        business_dict2[i] = business_id[i]
    business_profile_id = model.filter(lambda x: x['title'] == 'bpfile').map(lambda x: (x['business_index'], x['business_profile'])).collect()
    business_profile = {}
    for i in range(len(business_profile_id)):
        business_profile[business_profile_id[i][0]] = business_profile_id[i][1]    
    user_profile_id = model.filter(lambda x: x['title'] == 'upfile').map(lambda x: (x['user_index'], x['user_profile'])).collect()
    user_profile = {}
    for i in range(len(user_profile_id)):
        user_profile[user_profile_id[i][0]] = user_profile_id[i][1]    

    test_input = sc.textFile(input_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id']))
    test_input = test_input.map(lambda x: (user_dict.get(x[0], 6666666), business_dict.get(x[1], 6666666))).filter(lambda x: x[0] != 6666666 and x[1] != 6666666)
    prediction = test_input.map(lambda x: (x, cosSimilarity(user_profile.get(x[0], 6666666), business_profile.get(x[1], 6666666)))).collect()
    result = []
    for item in prediction:
        if item[1] > 0.01:
            line = {}
            line['user_id'] = user_dict2[item[0][0]]
            line['business_id'] = business_dict2[item[0][1]]
            line['sim'] = item[1]
            result.append(line)
    # output
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.write(str(json.dumps(line))+'\n')
        output_file.close()
    print('Duration:', (time.time()-start_time))
