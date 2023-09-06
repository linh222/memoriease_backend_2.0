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
schema = {
    "mappings": {
        "properties": {
            "ImageID": {"type": "text"},
            "new_name": {"type": "text"},
            "city": {"type": "text"},
            "event_id": {"type": "text"},
            "local_time": {"type": "date"},
            "semantic_name": {"type": "text"},
            "date_of_week": {"type": "text"},
            "time_period": {"type": "text"},
            "visual_concept": {"type": "text"},
            "blip_embed": {"type": "dense_vector", "dims": 256,
                           "index": True, "similarity": "cosine"
                           }
        }
    }
}
es.indices.create(index='lsc23', body=schema)

df = pd.read_json('{}/elasticsearch-data/data_4_ingest_visualconcept.json'.format(root_path), orient='index')
df['index'] = df['ImageID']


def gen_data():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        data.pop('index', None)
        yield {
            "_index": 'lsc23',
            "_id": row['index'],
            "_source": data
        }


bulk(es, gen_data())
