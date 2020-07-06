#%%
import sys
import json
from operator import add
from pyspark import SparkContext
if __name__ == '__main__':
    # review_file_path = "data/review_sample.json"
    # business_file_path = "data/business_sample.json"
    # output_file_path = "output2.json"
    # if_spark = "spark"  # either "spark" or "no_spark"
    # n = '6'
    review_file_path = sys.argv[1]
    business_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    if_spark = sys.argv[4]
    n = sys.argv[5]
    if if_spark == 'spark':
        sc = SparkContext.getOrCreate()
        review = sc.textFile(review_file_path).map(lambda x: json.loads(x))
        business = sc.textFile(business_file_path).map(lambda x: json.loads(x))
        business_star = review.map(lambda x: (x['business_id'], x['stars']))
        business_cate = business.map(lambda x: (x['business_id'], x['categories'])).filter(lambda x: (x[1] is not None) and (x[1] is not ''))
        star = business_star.groupByKey().map(lambda x: (x[0], (sum(x[1]), len(x[1]))))
        category = business_cate.filter(lambda x: (x[1] is not None) and (x[1] is not '')).mapValues(lambda x: [x.strip() for x in x.split(',')])
        joined = category.leftOuterJoin(star)
        temp = joined.map(lambda x: x[1]).filter(lambda x: (x[1] is not None) and (x[1] is not '')).flatMap(lambda x: [(cate, x[1]) for cate in x[0]])
        def addup(x,y):
            return (x[0]+y[0], x[1]+y[1])
        temp_groupBy_cate = temp.reduceByKey(addup)
        result = temp_groupBy_cate.map(lambda x: (x[0], float(x[1][0]/x[1][1])))
        def getValue(item):
            return (-item[1],item[0])
        ans = sorted(result.collect(), key=getValue)[:int(n)]

        
    else:
        review = open(review_file_path, encoding='utf8').readlines()
        business_star = list(map(lambda x: {"business_id": json.loads(x)["business_id"],
                                  'stars': json.loads(x)['stars']}, review))
        business = open(business_file_path, encoding='utf8').readlines()
        business_cate = list(map(lambda x: {"business_id": json.loads(x)["business_id"],
                                  'categories': json.loads(x)['categories']}, business))
        print('data_loaded')
        # business_star = []
        # for line in review:
        #     temp_dict = {}
        #     temp_dict['business_id'] = json.loads(line)['business_id']
        #     temp_dict['stars'] = json.loads(line)['stars']
        #     business_star.append(temp_dict)
        # business_cate = []
        # for line in business:
        #     temp_dict = {}
        #     temp_dict['business_id'] = json.loads(line)['business_id']
        #     temp_dict['categories'] = json.loads(line)['categories']
        #     business_cate.append(temp_dict)
        
        star = {}     
        def addup_star(x):
            value_add = star.get(x['business_id'])[0] + x['stars']
            count_add = star.get(x['business_id'])[1] + 1
            return (value_add, count_add)  
        for item in business_star:
            if item['business_id'] not in star.keys():                
                star[item['business_id']]=(float(item['stars']), 1)
            else:                
                star.update({item['business_id']: addup_star(item)})

        category = {}
        for item in business_cate:
            if (item['categories'] is not None) and (item['categories'] is not ''):
                category[item['business_id']]=[x.strip() for x in item['categories'].split(',')]
        print('dict_created')
        
        def addup_join(x):
            value_add = joined.get(x)[0] + float(star_count_tuple[0])
            count_add = joined.get(x)[1] + int(star_count_tuple[1])
            return (value_add, count_add) 
        joined = {}
        for item, star_count_tuple in star.items():
            if category.get(item) is not None:           
                for cates in category.get(item):
                    if cates not in joined.keys():                            
                        joined[cates] = star_count_tuple
                    else:
                        joined.update({cates: addup_join(cates)})
        combined = []
        for item in joined:
            combined.append((item, round((joined[item][0]/joined[item][1]), 2)))        
        print('calculate_finished')

        def getValue(item):
            return (-item[1],item[0])
        ans = sorted(combined, key=getValue)[:int(n)]
    result = {"result": ans}
    with open(output_file_path, 'w+') as output_file:
        json.dump(result, output_file)
    output_file.close()
# %%
