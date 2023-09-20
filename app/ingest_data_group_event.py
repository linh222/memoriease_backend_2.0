from os.path import join, dirname

import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

from config import HOST, root_path

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

es = Elasticsearch(hosts=[HOST], timeout=100)
# Ingest for group
schema = {
    "mappings": {
        "properties": {
            "group_id": {"type": "integer"},
            "embedding": {"type": "dense_vector", "dims": 256,
                          "index": True, "similarity": "cosine"
                          }
        }
    }
}

es.indices.create(index='linh_lsc23_groups_jan2020', body=schema)

df = pd.read_json(
    '{}/elasticsearch-data/group_indice.json'.format(root_path),
    orient='index')
df['index'] = df['group_id']


def gen_data():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        data.pop('index', None)
        yield {
            "_index": 'linh_lsc23_groups_jan2020',
            "_id": row['index'],
            "_source": data
        }


bulk(es, gen_data())

# Ingest for main event jan 2020
schema = {
    "mappings": {
        "properties": {
            "event": {"type": "integer"},
            "group": {"type": "integer"},
            "similar_image": {"type": "text"},
            "ImageID": {"type": "text"},
            "Tags": {"type": "text"},
            "ORC": {"type": "text"},
            "Caption": {"type": "text"},
            "minute_id": {"type": "text"},
            "new_name": {"type": "text"},
            "city": {"type": "text"},
            "local_time": {"type": "date"},
            "date_of_week": {"type": "text"},
            "time_period": {"type": "text"},
            "blip_embed": {"type": "dense_vector", "dims": 256,
                           "index": True, "similarity": "cosine"
                           }
        }
    }
}

es.indices.create(index='linh_lsc23_events_jan2020', body=schema)

df = pd.read_json(
    '{}/elasticsearch-data/event_indice.json'.format(root_path), orient='index')
df['index'] = df['ImageID']


def gen_data():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        data.pop('index', None)
        yield {
            "_index": 'linh_lsc23_events_jan2020',
            "_id": row['index'],
            "_source": data
        }


bulk(es, gen_data())
