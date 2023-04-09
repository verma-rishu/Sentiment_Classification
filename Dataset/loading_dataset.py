import pandas as pd
import json

def load_yelp_orig_data():
    PATH_TO_YELP_REVIEWS = 'yelp_academic_dataset_review.json'

    lines = []
    # read the entire file into a python array
    with open(PATH_TO_YELP_REVIEWS, 'r') as f:
        for line in f:
            lines.append(line)
    data = lines
    
    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)
    data_df.to_csv('output_reviews_top.csv')

def load_dataset():
    PATH_TO_YELP_REVIEWS = 'yelp_academic_dataset_review.json'
    with open(PATH_TO_YELP_REVIEWS) as json_file:
        write_header = True
        for line in json_file:
            if not line.strip:
                continue
            data = json.loads(line)
            df = pd.json_normalize(data)
            df.to_csv('yadr_chunks.csv', mode='a', index=False, header=write_header)
            write_header = False

load_dataset()