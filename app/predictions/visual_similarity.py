import numpy as np
from app.config import embed_directory, HOST, INDICES
import json
import logging
from app.predictions.utils import process_query, construct_filter, send_request_to_elasticsearch, extract_advanced_filter, add_advanced_filters
from app.predictions.predict import retrieve_image

logging.basicConfig(filename='memoriease_backend.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_img_and_emb(image_id, directory):
    # Get the image embedding
    year = image_id[:6]
    day = image_id[6:8]
    image_id = image_id.replace('.jpg', '')
    emb_path = f'{directory}/{year}/{day}/{image_id}.npy'
    emb = np.load(emb_path)
    return emb


def calculate_mean_emb(image_id):
    # Aggregate the input images by mean
    list_embed = []
    for image in image_id:
        emb = load_img_and_emb(image, embed_directory)
        list_embed.append(emb)
    avg_embed = sum(list_embed) / len(list_embed)
    return avg_embed


def relevance_image_similar(image_embedding, query, image_id=None, size=100):
    if image_id is None:
        image_id = []
    col = ["day_of_week", "ImageID", "local_time", "new_name", 'event_id']
    returned_query, advanced_filters = extract_advanced_filter(query)
    logging.info(f"Visual similarity: Extracted advanced search: {advanced_filters}")
    processed_query, list_keyword, time_period, weekday, time_filter, location = process_query(query)
    query_dict = {
        "time_period": time_period,
        "location": location,
        "list_keyword": list_keyword,
        "weekday": weekday,
        "time_filter": time_filter,
        "semantic_name": ''
    }
    if len(advanced_filters) > 0:
        query_dict = add_advanced_filters(advanced_filters, query_dict)
    logging.info(f"Visual similarity: Query dictionary: {query_dict}")
    logging.info(f"Visual similarity: query dict: {query_dict}")
    if len(image_id) > 0:
        query_dict['image_excluded'] = image_id
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


def pseudo_relevance_feedback(concept_query: str, embed_model, txt_processor, top_k=3):
    # Retrieve top 3 and assume it as true results to get more similar images to top 3
    initial_result = retrieve_image(concept_query=concept_query, embed_model=embed_model, txt_processor=txt_processor)
    top_k_result = initial_result['hits']['hits'][0:top_k]
    relevant_image_id = [i['_id'][:-4] for i in top_k_result]
    relevant_embedding = calculate_mean_emb(relevant_image_id)
    result = relevance_image_similar(relevant_embedding, query=concept_query, size=100)
    return result
