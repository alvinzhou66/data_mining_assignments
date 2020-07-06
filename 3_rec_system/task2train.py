import collections
import json
import math
import re
import string
import sys
import time
from pyspark import SparkConf, SparkContext
def writeModelList(input_list, keyword):
    result = []
    for x in input_list:
        for key, value in x.items():
            result.append({keyword[0]: key, keyword[1]: value})
    return result
def writeModelDict(input_dict, keyword):
    result = []
    for key, value in input_dict.items():
        result.append({keyword[0]: key, keyword[1]: value})
    return result
def removePunctuations(text, stopwords):
    result = []
    for t in list(text):
        t = re.sub('[%s]' % re.escape(string.punctuation+string.digits), '', t)
        result.extend(list(filter(lambda x: x not in stopwords and x not in string.ascii_lowercase and x != '', re.split(r"[~\r\n\s]", t))))
    return result
def wordCount(text):
    word_dict = collections.defaultdict(int)
    most_frequent = 0
    for word in text:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    most_frequent = max(max(word_dict.values()), most_frequent)
    word_dict = dict(filter(lambda x: x[1] > 3, word_dict.items()))
    result = [(key, value, most_frequent) for key, value in word_dict.items()]
    result = sorted(result, key=lambda x: x[1], reverse = True)
    return result
def pickTop200(tf_idf_scores):
    scores = sorted(list(tf_idf_scores), reverse=True, key=lambda x: x[1])
    result = scores[:200]
    return result
def convertToIndex(input_list):
    result = []
    for item in input_list:
        if item in business_profile_dict.keys():
            result.extend(list(business_profile_dict[item]))
    return result
if __name__ == '__main__':
    start_time = time.time()
    # define input variables
    input_file_path = "data/train_review.json"
    output_file_path = "task2.model"
    stopwords_file_path = "data/stopwords"
    # input_file_path = sys.argv[1]
    # output_file_path = sys.argv[2]
    # stopwords_file_path = sys.argv[3]
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)    
    
    stopwords = set(word.strip() for word in open(stopwords_file_path))
    prediction_model = []

    review = sc.textFile(input_file_path).map(lambda x: json.loads(x))
    
    user_id = review.map(lambda x: x['user_id']).distinct().sortBy(lambda x: x).collect()
    user_dict = {}
    for i in range(len(user_id)):
        user_dict[user_id[i]] = i
    
    business_id = review.map(lambda x: x['business_id']).distinct().sortBy(lambda x: x).collect()
    business_dict = {}
    for i in range(len(business_id)):
        business_dict[business_id[i]] = i
    
    business_tf = review.map(lambda x: (business_dict[x['business_id']], str(x['text'].encode('utf-8')).lower())).groupByKey()

    business_tf = business_tf.mapValues(lambda x: removePunctuations(x, stopwords)).mapValues(lambda x: wordCount(x)).flatMap(lambda x: [(x[0], y[0], y[1]/y[2]) for y in x[1]])
    
    business_idf = business_tf.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: list(set(x)))
    business_idf = business_idf.flatMap(lambda x: [((doc, x[0]), math.log2(len(business_dict)/len(x[1]))) for doc in x[1]])

    tf_idf = business_tf.map(lambda x: ((x[0], x[1]), x[2])).leftOuterJoin(business_idf)
    tf_idf = tf_idf.mapValues(lambda x: x[0]*x[1]).map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey()

    tf_idf = tf_idf.mapValues(lambda x: pickTop200(x))

    word_id = tf_idf.flatMap(lambda x: [(y[0], 1) for y in x[1]]).groupByKey().map(lambda x: x[0]).collect()
    word_dict = {}
    for i in range(len(word_id)):
        word_dict[word_id[i]] = i
    
    business_profile = tf_idf.mapValues(lambda x: [word_dict[y[0]] for y in x]).map(lambda x: {x[0]: x[1]})
    business_profile_dict = {}
    for item in business_profile.collect():
        (key, value), = item.items()
        business_profile_dict[key] = value
    
    user_profile_temp = review.map(lambda x: (x['user_id'], x['business_id'])).groupByKey().map(lambda x: (user_dict[x[0]], list(set(x[1]))))

    user_profile_temp = user_profile_temp.mapValues(lambda x: [business_dict[y] for y in x]).collect()
    user_profile = {}
    for user in user_profile_temp:
        value = convertToIndex(user[1])
        if len(value) > 1:
            user_profile[user[0]] = list(set(value))   

    prediction_model.extend(writeModelList(business_profile.collect(), ['business_index', 'business_profile']))
    prediction_model.extend(writeModelDict(user_profile, ['user_index', 'user_profile']))
    # output
    result = prediction_model
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.write(str(json.dumps(line))+'\n')
        output_file.close()
    print('Duration:', (time.time()-start_time))

    