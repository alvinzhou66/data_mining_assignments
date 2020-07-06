#%%
import csv
import json
from pyspark import SparkContext
input_review_file_path = "data/review.json"
input_business_file_path = "data/business.json"
output_file_path = "data/user_business.csv"
sc = SparkContext.getOrCreate()
business = sc.textFile(input_business_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['state']))
business_id = business.filter(lambda x: x[1]=='NV').map(lambda x: x[0]).collect()
review = sc.textFile(input_review_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id']))
user_id = review.filter(lambda x: x[1] in business_id).collect()
with open(output_file_path, 'w+', newline='') as output_file:
    output_line = csv.writer(output_file)
    output_line.writerow(['user_id', 'business_id'])
    for line in user_id:
        output_line.writerow(line)
    output_file.close()

# %%
