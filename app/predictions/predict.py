from app.predictions.utils import process_query, construct_filter, build_query_template
import requests
import json
import torch
from elasticsearch import Elasticsearch

from app.config import settings, HOST, root_path, INDICES
from app.predictions.blip_extractor import extract_query_blip_embedding
import json
from LAVIS.lavis.models import load_model_and_preprocess
import requests


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

    query_dict = {
        "time_period": time_period,
        "location": location,
        "list_keyword": list_keyword,
        "weekday": weekday,
        "time_filter": time_filter,
        "semantic_name": semantic_name,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "is_weekend": is_weekend,
    }

    filter, must = construct_filter(query_dict)

    query_template = build_query_template(filter, must, text_embedding, size=100)
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
    tree at our rented house in Thailand.""", embed_model=model, txt_processor=txt_processors)
    with open('{}/app/evaluation_model/result.json'.format(root_path), 'w') as f:
        json.dump(result['hits']['hits'], f)
