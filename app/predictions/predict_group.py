import json

import requests

from app.config import HOST, INDICES, GROUP_INDICES, root_path
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.utils import process_query, construct_filter, build_query_template
import torch
from LAVIS1.lavis.models import load_model_and_preprocess

def retrieve_image(
        concept_query: str,
        embed_model,
        txt_processor,
        semantic_name="",
        start_hour="",
        end_hour="",
        is_weekend="",
):
    processed_query, list_keyword, time_period, weekday, time_filter, location = process_query(concept_query)
    text_embedding = extract_query_blip_embedding(processed_query, embed_model, txt_processor)
    # Search for group
    query_group = {

        "knn": {
            "field": "embedding",
            "query_vector": text_embedding.tolist(),
            "k": 10,
            "num_candidates": 1000,
        },

        "_source": ['group_id'],
        "size": 10,
    }

    query_group = json.dumps(query_group)
    url = f"{HOST}/{GROUP_INDICES}/_search"

    with requests.Session() as session:
        try:
            response = session.post(url, data=query_group, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            ValueError(e)
            return None
    list_groups = [i['_source']['group_id'] for i in result['hits']['hits']]


    query_dict = {
        "time_period": time_period,
        "location": location,
        "list_keyword": list_keyword,
        "weekday": weekday,
        "time_filter": time_filter,
        "semantic_name": semantic_name if semantic_name is not None else '',
        "start_hour": start_hour if start_hour is not None else '',
        "end_hour": end_hour if end_hour is not None else '',
        "is_weekend": is_weekend if is_weekend is not None else '',
        # "groups": list_groups if list_groups is not None else '',
    }

    filter, must = construct_filter(query_dict)

    query_template = build_query_template(filter, must, text_embedding, size=100)
    print(query_template)
    query_template = json.dumps(query_template)
    url = f"{HOST}/{INDICES}/_search"

    with requests.Session() as session:
        try:
            response = session.post(url, data=query_template, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            ValueError(e)
            return None

if __name__ == "__main__":
    # Remote server
    import time

    start_time = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                      model_type="coco", is_eval=True,
                                                                      device=device)
    print("cuda" if torch.cuda.is_available() else "cpu")
    print(time.time() - start_time, 'seconds')

    result = retrieve_image(concept_query="""Exotic birds. Find examples of multicoloured parrots (real or fake) in a
    tree at our rented house""", embed_model=model, txt_processor=txt_processors)
    with open('{}/app/evaluation_model/result_group_event.json'.format(root_path), 'w') as f:
        json.dump(result, f)
