import torch
from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, status
from fastapi.openapi.models import APIKey

from LAVIS.lavis.models import load_model_and_preprocess
from app.api_key import get_api_key
from app.apis.api_utils import add_image_link, RequestTimestampMiddleware
from app.config import HOST
from app.predictions.predict import retrieve_image
from app.predictions.question_answering import process_result
from app.predictions.temporal_predict import temporal_search
from app.predictions.utils import automatic_logging
from .schemas import (
    FeatureModelSingleSearch,
    FeatureModelTemporalSearch
)

router = APIRouter()


# Function to initialize resources
def initialize_resources():
    # Load resource in the start
    global es, model, vis_processors, txt_processor, logger
    global instruct_model, instruct_vis_processor, instruct_txt_processor, device

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processor = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
    )
    instruct_model, instruct_vis_processor, instruct_txt_processor = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device
    )

    print('Loading 2 models successfully at ', device)


@router.post(
    "/predict_temporal",
    status_code=status.HTTP_200_OK,
)
async def predict_image_temporal(feature: FeatureModelTemporalSearch, api_key: APIKey = Depends(get_api_key)):
    # Predict temporal
    # Input: before, main, after event, filters
    # Output: list of dicts with three keys: current_event, previous_event, after_event
    query = feature.query
    semantic_name = feature.semantic_name

    # Perform search
    results = temporal_search(concept_query=query, embed_model=model, txt_processor=txt_processor,
                              previous_event=feature.previous_event,
                              next_event=feature.next_event, time_gap=feature.time_gap, semantic_name=semantic_name)
    results = add_image_link(results)

    # Automatic run Logging query string
    # automatic_logging(results=results, output_file_name='ntcir_automatic_logging')
    return results


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
)
async def predict_image(feature: FeatureModelSingleSearch, api_key: APIKey = Depends(get_api_key)):
    # Predict single moment
    # Input: query, filters
    # Output: list of dicts with 1 keys: current_event
    query = feature.query
    topic = feature.topic
    semantic_name = feature.semantic_name
    start_hour = feature.start_hour
    end_hour = feature.end_hour
    is_weekend = feature.is_weekend

    # Perform search
    raw_result = retrieve_image(concept_query=query, embed_model=model, txt_processor=txt_processor,
                                semantic_name=semantic_name, start_hour=start_hour,
                                end_hour=end_hour, is_weekend=is_weekend)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)

    # Automatic run Logging query string
    # automatic_logging(results=results, output_file_name='ntcir_automatic_logging')

    return results


@router.post(
    "/question_answering",
    status_code=status.HTTP_200_OK,
)
async def question_answering(feature: FeatureModelSingleSearch, api_key: APIKey = Depends(get_api_key)):
    # Question answering endpoint, give the question and some filters if possible
    # Output: dictionary with key: question.
    query = feature.query
    topic = feature.topic
    semantic_name = feature.semantic_name
    start_hour = feature.start_hour
    end_hour = feature.end_hour
    is_weekend = feature.is_weekend

    answer = process_result(query=query, blip2_embed_model=model, blip2_txt_processor=txt_processor,
                            instruct_model=instruct_model, instruct_vis_processor=instruct_vis_processor,
                            device=device, semantic_name=semantic_name, start_hour=start_hour,
                            end_hour=end_hour, is_weekend=is_weekend)

    return {"answer": answer}


def include_router(app):
    app.include_router(router)
    app.add_middleware(RequestTimestampMiddleware, router_path='/predict')


initialize_resources()
