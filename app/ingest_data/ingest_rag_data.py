from io import StringIO
from os.path import join, dirname
from tqdm import tqdm
import boto3
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import HOST, AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET, RAG_INDICES
from utils import create_index

dotenv_path = join(dirname(__file__), '../../.env')
load_dotenv(dotenv_path)

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
es = Elasticsearch(hosts=[HOST], timeout=100)
schema = {
    "mappings": {
        "properties": {
            "event_id": {"type": "text"},
            "description": {"type": "text"},
            "ImageID": {"type": "text"},
            "new_name": {"type": "text"},
            "city": {"type": "text"},
            "local_time": {"type": "date"},
            "day_of_week": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 768,
                          "index": True, "similarity": "cosine"
                          }
        }
    }
}
if es.indices.exists(index=RAG_INDICES):
    es.indices.delete(index=RAG_INDICES)
result = create_index(es=es, indice=RAG_INDICES, schema=schema)

response = s3.get_object(Bucket=BUCKET, Key='lsc24_rag_description.json')
json_data = response['Body'].read().decode('utf-8')
df = pd.read_json(
    StringIO(json_data),
    orient='index')


# index_data2elasticsearch(df=df, indice=INDICES, host=HOST)
def gen_data():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        row['local_time'] = pd.to_datetime(row['local_time']).isoformat()
        data = row.to_dict()
        data.pop('index', None)
        yield {
            "_index": RAG_INDICES,
            "_id": row['ImageID'],
            "_source": data
        }


bulk(es, gen_data())
