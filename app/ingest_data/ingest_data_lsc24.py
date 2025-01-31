from io import StringIO
from os.path import join, dirname
from tqdm import tqdm
import boto3
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import HOST, AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET, INDICES
from utils import create_index

dotenv_path = join(dirname(__file__), '../../.env')
load_dotenv(dotenv_path)

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
es = Elasticsearch(hosts=[HOST], timeout=1000)
schema = {
    "mappings": {
        "properties": {
            "ImageID": {"type": "text"},
            "Tags": {"type": "text"},
            "ORC": {"type": "text"},
            "Caption": {"type": "text"},
            "new_name": {"type": "text"},
            "city": {"type": "text"},
            "event_id": {"type": "text"},
            "local_time": {"type": "date"},
            "semantic_name": {"type": "text"},
            "hour": {"type": "integer"},
            "date_of_week": {"type": "text"},
            "is_weekend": {"type": "integer"},
            "time_period": {"type": "text"},
            "blip_embed": {"type": "dense_vector", "dims": 256,
                           "index": True, "similarity": "cosine"
                           },
            "description": {"type": "text"}
        }
    }
}
if es.indices.exists(index=INDICES):
    es.indices.delete(index=INDICES)
result = create_index(es=es, indice=INDICES, schema=schema)

response = s3.get_object(Bucket=BUCKET, Key='grouped_info_dict_full_blip2_lsc24.json')
json_data = response['Body'].read().decode('utf-8')
df = pd.read_json(
    StringIO(json_data),
    orient='index')

#index_data2elasticsearch(df=df, indice=INDICES, host=HOST)
def gen_data():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        data.pop('index', None)
        yield {
            "_index": INDICES,
            "_id": row['ImageID'],
            "_source": data
        }


bulk(es, gen_data())

