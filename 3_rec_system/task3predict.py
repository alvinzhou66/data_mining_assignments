import json
import sys
import time
from pyspark import SparkConf, SparkContext


if __name__ == '__main__':
    start_time = time.time()
    # train_file_path = "data/train_review2.json"
    # test_file_path = "data/test_review.json"
    # model_file_path = "task3user.model"
    # output_file_path = "task3user.predict"
    # model_type = "user_based"  # either "item_based" or "user_based"
    # bus_avg_file_path = "data/business_avg.json"
    # user_avg_file_path = "data/user_avg.json"

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    model_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    model_type = sys.argv[5]
    bus_avg_file_path = "$/ASNLIB/publicdata/business_avg.json"
    user_avg_file_path = "$/ASNLIB/publicdata/user_avg.json"

    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    review= sc.textFile(train_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'], x['stars']))
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
        business_pair = sc.textFile(model_file_path).map(lambda x: json.loads(x)).collect()
        bpair_dict = {}
        for i in range(len(business_pair)):
            bpair_dict[(business_dict[business_pair[i]['b1']], business_dict[business_pair[i]['b2']])] = business_pair[i]['sim']
        
        train_ubs = review.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()
        train_ubs = train_ubs.mapValues(lambda x: [(business_dict[y[0]], y[1]) for y in list(set(x))]).map(lambda x: (user_dict[x[0]], x[1]))
        bavg_dict = sc.textFile(bus_avg_file_path).map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()).collectAsMap()
        test_input = sc.textFile(test_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id']))
        test_input = test_input.map(lambda x: (user_dict.get(x[0], 6666666), business_dict.get(x[1], 6666666))).filter(lambda x: x[0] != 6666666 and x[1] != 6666666).leftOuterJoin(train_ubs)
        def item_based_prediction(input_list):
            business = input_list[0]
            bs_list_forThisUser = list(input_list[1])
            temp = []
            for item in bs_list_forThisUser:
                pair = tuple(sorted([int(item[0]), business]))
                temp.append((item[1], bpair_dict.get(pair, 0)))
            neighbors = sorted(temp, key=lambda x: x[1], reverse=True)[:3]
            above = 0
            for item in neighbors:
                above += item[0]*item[1]
            below = 0
            for item in neighbors:
                below += abs(item[1])
            if above != 0 and below != 0:
                return above/below
            else:
                return bavg_dict.get(business_dict2.get(business, 6666666), 3.8)
        compute_similarity = test_input.mapValues(lambda x: (x[0], item_based_prediction(x))).collect()
        for item in compute_similarity:
            line = {}
            line['user_id'] = user_dict2[item[0]]
            line['business_id'] = business_dict2[item[1][0]]
            line['stars'] = item[1][1]
            prediction_model.append(line)
    elif model_type == 'user_based':
        prediction_model = []
        user_pair = sc.textFile(model_file_path).map(lambda x: json.loads(x)).collect()
        upair_dict = {}
        for i in range(len(user_pair)):
            upair_dict[(user_dict[user_pair[i]['u1']], user_dict[user_pair[i]['u2']])] = user_pair[i]['sim']
        
        train_bus = review.map(lambda x: (x[1], (x[0], x[2]))).groupByKey()
        train_bus = train_bus.map(lambda x: (business_dict[x[0]], x[1])).mapValues(lambda x: [(user_dict[y[0]], y[1]) for y in list(set(x))])
        uavg_dict = sc.textFile(user_avg_file_path).map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()).collectAsMap()
        test_input = sc.textFile(test_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['user_id']))
        test_input = test_input.map(lambda x: (business_dict.get(x[0], 6666666), user_dict.get(x[1], 6666666))).filter(lambda x: x[0] != 6666666 and x[1] != 6666666).leftOuterJoin(train_bus)
        def user_based_prediction(input_list):
            user = input_list[0]
            u_list_forThisBus = list(input_list[1])
            temp = []
            for item in u_list_forThisBus:
                pair = tuple(sorted([int(item[0]), user]))
                ub_set = user_dict2.get(item[0], 6666666)
                avg_star = uavg_dict.get(ub_set, 3.8)
                temp.append((item[1], avg_star, upair_dict.get(pair, 0)))
            
            above = 0
            for item in temp:
                above += (item[0]-item[1])*item[2]
            below = 0
            for item in temp:
                below += abs(item[2])
            if above != 0 and below != 0:
                return above/below + uavg_dict.get(user_dict2.get(user, 6666666), 3.8)
            else:
                return uavg_dict.get(user_dict2.get(user, 6666666), 3.8)
        compute_similarity = test_input.mapValues(lambda x: (x[0], user_based_prediction(x))).collect()
        for item in compute_similarity:
            line = {}
            line['user_id'] = user_dict2[item[1][0]]
            line['business_id'] = business_dict2[item[0]]
            line['stars'] = item[1][1]
            prediction_model.append(line)
    result = prediction_model
    # output
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.write(str(json.dumps(line))+'\n')
        output_file.close()
    print('Duration:', (time.time()-start_time))


