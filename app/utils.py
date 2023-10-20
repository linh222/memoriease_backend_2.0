import json
from os.path import join, dirname

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import HOST

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)


def row_to_dict(row):
    # Convert timestamps to ISO formatted strings
    row['local_time'] = row['local_time'].isoformat()
    return row.to_dict()


def create_index(es, indice, schema):
    response = es.indices.create(index=indice, body=schema)
    return response


def index_data2elasticsearch(df, indice, host=HOST):
    for i in tqdm(range(df.shape[0])):
        data = row_to_dict(df.iloc[i, :])
        json_data = json.dumps(data)
        url = f"{host}/{indice}/_doc/{i}"

        # Send a POST request to index the document
        response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

        # Check the response
        if response.status_code != 201:
            ValueError(f"Fail for row: {i}")

# if __name__ == "__main__":
#     schema = {
#         "mappings": {
#             "properties": {
#                 "ImageID": {"type": "text"},
#                 "Tags": {"type": "text"},
#                 "ORC": {"type": "text"},
#                 "Caption": {"type": "text"},
#                 "new_name": {"type": "text"},
#                 "city": {"type": "text"},
#                 "event_id": {"type": "text"},
#                 "local_time": {"type": "date"},
#                 "semantic_name": {"type": "text"},
#                 "hour": {"type": "integer"},
#                 "date_of_week": {"type": "text"},
#                 "is_weekend": {"type": "integer"},
#                 "time_period": {"type": "text"},
#                 "blip_embed": {"type": "dense_vector", "dims": 256,
#                                "index": True, "similarity": "cosine"
#                                }
#             }
#         }
#     }
# create_index(HOST, 'new_indice', schema)
