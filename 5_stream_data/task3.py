import tweepy
import csv
import sys
import collections
from random import sample
access_token = '1149215042316857344-rQ3lC2D2ATSVPrnI48FF3BOK7rX4Dh'
access_token_secret = '8hOng9ZXeitNX29GyRW7dWaEyZqnHEIEwvPgHXf4w5urG'
api_key = 'Mh1pmTZHJiZQF6LQkoyDN1FPP'
api_secret = 'sR7sPWuRl0pf2DAV3bNdHz0LqFzI0s2MRPTgZpKpS6zUFhl8rd'
topic = ['NBA','Basketball', 'Lakers']
def addTags(tag_list, new_tag_list):
    for new_tag in new_tag_list:
        if sample(range(100), 1) == 42:
            tag_list.pop(sample(range(100),1))
            tag_list.append(new_tag)
    return tag_list
class Tweepy(tweepy.StreamListener):
    def __init__(self, output_file_path):
        tweepy.StreamListener.__init__(self)
        self.tweet_sequence = 0
        self.tag_list = []
        self.top3_tag = []
        self.output_file_path = output_file_path
        with open(self.output_file_path, 'w') as output_file:
            csv.writer(output_file)
    def on_status(self, status):
        tag_dict = status.entities.get('hashtags')
        if len(tag_dict) > 0:
            new_tag_list = [item.get('text') for item in tag_dict]
            self.tweet_sequence += 1
            if len(self.tag_list)+len(new_tag_list) <= 100:
                self.tag_list.extend(new_tag_list)
            elif len(self.tag_list)+len(new_tag_list) > 100 and len(self.tag_list) < 100:
                self.tag_list.extend(new_tag_list[:100-len(self.tag_list)])
                addTags(self.tag_list, new_tag_list[100-len(self.tag_list):])
            else:
                addTags(self.tag_list, new_tag_list)
            tag_count0 = collections.Counter(self.tag_list)
            tag_count0 = list(dict(tag_count0).items())
            # print(tag_count0)           
            tag_count = sorted(tag_count0, key = lambda x: (-x[1], x[0]))
            third_count = -1
            current_sequence = 4
            self.top3_tag = []
            for tag, count in tag_count:
                if count != third_count:
                    third_count = count
                    current_sequence -= 1
                if current_sequence > 0:
                    self.top3_tag.append((tag, count))
                else:
                    break
            output_file = open(self.output_file_path, 'a+', encoding='utf-8')
            output_file.write("The number of tweets with tags from the beginning: {}\n".format(self.tweet_sequence))
            print(self.top3_tag)
            for item in self.top3_tag:
                output_file.write(item[0]+':'+str(item[1])+'\n')
            output_file.write('\n')
            output_file.close()
if __name__ == '__main__':
    port = sys.argv[1]
    output_file_path = sys.argv[2]
    # port = 9999
    # output_file_path = 'task3.csv'
    listener = Tweepy(output_file_path)
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = tweepy.Stream(auth=auth, listener=listener)
    stream.filter(track=topic,languages=["en"])