import numpy as np
from app.config import settings, HOST, INDICES
import json
import requests
from app.predictions.utils import process_query, construct_filter, send_request_to_elasticsearch
from app.predictions.predict import retrieve_image


def load_img_and_emb(image_id, directory):
    # Get the image embedding
    year = image_id[:6]
    day = image_id[6:8]
    emb_path = f'{directory}/{year}/{day}/{image_id}.npy'
    emb = np.load(emb_path)
    return emb


def calculate_mean_emb(image_id):
    # Aggregate the input images by mean
    list_embed = []
    for image in image_id:
        emb = load_img_and_emb(image, settings.embed_directory)
        list_embed.append(emb)
    avg_embed = sum(list_embed) / len(list_embed)
    return avg_embed


def relevance_image_similar(image_embedding, query, semantic_name, size=100):
    col = ["day_of_week", "ImageID", "local_time", "new_name", 'event_id']
    processed_query, list_keyword, time_period, weekday, time_filter, location = process_query(query)
    query_dict = {
        'time_period': time_period,
        'location': location,
        'list_keyword': list_keyword,
        'weekday': weekday,
        'time_filter': time_filter,
        'semantic_name': semantic_name if semantic_name is not None else ''
    }
    filters = construct_filter(query_dict)
    query_template = {

        "knn": {
            "field": "blip_embed",
            "query_vector": image_embedding.tolist(),
            "k": size,
            "num_candidates": 1000,
            "filter": filters
        },
        "_source": col,
        "size": size,
    }

    query_template = json.dumps(query_template)
    results = send_request_to_elasticsearch(HOST, INDICES, query_template)
    return results


def pseudo_relevance_feedback(concept_query: str, embed_model, txt_processor, semantic_name=None,
                              start_hour=None, end_hour=None, is_weekend=None, top_k=3):
    # Retrieve top 3 and assume it as true results to get more similar images to top 3
    initial_result = retrieve_image(concept_query=concept_query, embed_model=embed_model, txt_processor=txt_processor)
    top_k_result = initial_result['hits']['hits'][0:top_k]
    relevant_image_id = [i['_id'][:-4] for i in top_k_result]
    relevant_embedding = calculate_mean_emb(relevant_image_id)
    result = relevance_image_similar(relevant_embedding, query=concept_query, semantic_name=semantic_name, size=100)
    return result
