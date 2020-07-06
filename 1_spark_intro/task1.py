#%%
import sys
import json
from operator import add
from pyspark import SparkContext

if __name__ == '__main__':
    # input_file_path = "data/review.json"
    # output_file_path = "output1.json"
    # stopwords_file_path = "data/stopwords"
    # year = '2018'
    # m = '2'
    # n = '10'
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    stopwords_file_path = sys.argv[3]
    year = sys.argv[4]
    m = sys.argv[5]
    n = sys.argv[6]
    punctuations = ['(',')','[',']',',','.','!','?',':',';','\n']
    stop_words = set(word.strip() for word in open(stopwords_file_path))
    sc = SparkContext.getOrCreate()
    review = sc.textFile(input_file_path).map(lambda x: json.loads(x))
    
    #A
    ans_a = review.map(lambda x: x['review_id']).count()
    print(ans_a)
    
    #B
    review_groupBy_idAndDate = review.map(lambda x: (x['review_id'], x['date']))
    ans_b = review_groupBy_idAndDate.filter(lambda x: x[1].split('-')[0] == year).count()
    print(ans_b)
    
    #C
    ans_c = review.map(lambda x: x['user_id']).distinct().count()
    print(ans_c)
    
    #D
    review_groupBy_user = review.map(lambda x: (x['user_id'], 1)).reduceByKey(add)
    def getValue(item):
        return item[1]
    ans_d = sorted(review_groupBy_user.collect(), key=getValue, reverse=True)[:int(m)]   
    print(ans_d)

    #E
    review_word = review.map(lambda x: x['text']).flatMap(lambda x: x.lower().split(' '))
    def ignore_punctuations(word):
        if word not in stop_words:
            for i in word:
                if i in punctuations:
                    word = word.replace(i,'')
            if len(word) != 0:
                return word
    review_word_without_punc = review_word.map(lambda x: (ignore_punctuations(x), 1)).filter(lambda x: x[0] is not None).reduceByKey(add)
    ans_e_temp = sorted(review_word_without_punc.collect(), key=getValue, reverse=True)[:int(n)]    
    ans_e = []
    for item in ans_e_temp:
        ans_e.append(item[0])
    print(ans_e)
    
    #output
    result = {}
    result['A'] = ans_a
    result['B'] = ans_b
    result['C'] = ans_c
    result['D'] = ans_d
    result['E'] = ans_e

    with open(output_file_path, 'w+') as output_file:
        json.dump(result, output_file)
    output_file.close()