import time

import pandas as pd
import torch
from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, status
from fastapi.openapi.models import APIKey

from LAVIS.lavis.models import load_model_and_preprocess
from app.api_key import get_api_key
from app.apis.api_utils import add_image_link, RequestTimestampMiddleware, metadata_logging2file
from app.config import HOST, root_path
from app.predictions.predict import retrieve_image
from app.predictions.relevance_feedback import relevance_image_similar, calculate_mean_emb, pseudo_relevance_feedback
from app.predictions.temporal_predict import temporal_search
from app.predictions.utils import automatic_logging
from .schemas import (
    FeatureModelRelevanceSearch,
    FeatureModelSingleSearch,
    FeatureModelTemporalSearch,
    FeatureModelSingleNTCIRSearch,
)

router = APIRouter()


# Function to initialize resources
def initialize_resources():
    global es, model, vis_processors, txt_processor, df_main_event, logger

    es = Elasticsearch(hosts=[HOST], timeout=100)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processor = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
    )

    print('Loading model successfully at')
    print("cuda" if torch.cuda.is_available() else "cpu")

    df_main_event = pd.read_csv('{}/app/models/event_segmentation_appoarch2_0_75.csv'.format(root_path))


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
)
async def predict_image(feature: FeatureModelSingleSearch, api_key: APIKey = Depends(get_api_key)):
    query = feature.query
    topic = feature.topic

    # Logging query submit
    metadata_logging2file(query, topic)

    raw_result = retrieve_image(concept_query=query, embed_model=model, txt_processor=txt_processor)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)

    # Logging list_receive
    response = [i['current_event']['_source']['ImageID'] for i in results]
    timestamp = int(time.time())
    with open("{}/app/evaluation_model/metadata_log.txt".format(root_path), "a") as file:
        file.write("\n" + "{},{},{},{}".format(timestamp, topic, 'list_received', response))

    return results


@router.post(
    "/predict_temporal",
    status_code=status.HTTP_200_OK,
)
async def predict_image_temporal(feature: FeatureModelTemporalSearch, api_key: APIKey = Depends(get_api_key)):
    query = feature.query
    semantic_name = feature.semantic_name

    results = temporal_search(concept_query=query, embed_model=model, txt_processor=txt_processor,
                              previous_event=feature.previous_event,
                              next_event=feature.next_event, time_gap=feature.time_gap, semantic_name=semantic_name)
    results = add_image_link(results)

    # Automatic run Logging query string
    automatic_logging(results=results, output_file_name='ntcir_automatic_logging')
    return results


@router.post(
    "/predict_ntcir",
    status_code=status.HTTP_200_OK,
)
async def predict_image(feature: FeatureModelSingleNTCIRSearch, api_key: APIKey = Depends(get_api_key)):
    query = feature.query
    semantic_name = feature.semantic_name

    raw_result = retrieve_image(concept_query=query, embed_model=model, txt_processor=txt_processor,
                                semantic_name=semantic_name, start_hour=feature.start_hour,
                                end_hour=feature.end_hour, is_weekend=feature.is_weekend)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)

    # Automatic run Logging query string
    automatic_logging(results=results, output_file_name='ntcir_automatic_logging')

    return results


@router.post(
    "/relevance_feedback",
    status_code=status.HTTP_200_OK,
)
async def relevance_feedback(feature: FeatureModelRelevanceSearch):
    query = feature.query
    image_id = feature.image_id
    semantic_name = feature.semantic_name
    mean_embedding = calculate_mean_emb(image_id=image_id)
    raw_result = relevance_image_similar(image_embedding=mean_embedding, query=query, semantic_name=semantic_name)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)
    return results


@router.post(
    "/predict_pseudo_relevance_feedback",
    status_code=status.HTTP_200_OK,
)
async def predict_image_peuso_rf(feature: FeatureModelSingleNTCIRSearch, api_key: APIKey = Depends(get_api_key)):
    query = feature.query
    semantic_name = feature.semantic_name

    raw_result = pseudo_relevance_feedback(concept_query=query, embed_model=model, txt_processor=txt_processor,
                                           semantic_name=semantic_name, start_hour=feature.start_hour,
                                           end_hour=feature.end_hour, is_weekend=feature.is_weekend, top_k=10)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)

    return results


def include_router(app):
    app.include_router(router)
    app.add_middleware(RequestTimestampMiddleware, router_path='/predict')


initialize_resources()
