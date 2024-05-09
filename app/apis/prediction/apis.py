import torch
from fastapi import APIRouter, Depends, status
from fastapi.openapi.models import APIKey
from sentence_transformers import SentenceTransformer

from LAVIS.lavis.models import load_model_and_preprocess
from app.api_key import get_api_key
from app.apis.api_utils import RequestTimestampMiddleware, add_image_link
from app.predictions.chat_conversation import chat
from app.predictions.temporal_predict import temporal_search
from .schemas import (
    FeatureModelSingleSearch,
    FeatureModelTemporalSearch,
    FeatureModelConversationalSearch,
    FeatureModelVisualSimilarity
)
import json
from app.config import HOST, INDICES, model_rag_path
from app.predictions.visual_similarity import relevance_image_similar, calculate_mean_emb
from app.predictions.utils import send_request_to_elasticsearch
from app.predictions.predict import retrieve_image
from app.predictions.rag_question_answering import rag_question_answering

router = APIRouter()


# Function to initialize resources
def initialize_resources():
    # Load resource in the start
    global es, model, vis_processors, txt_processor, logger
    global embedding_model, tokenizer, device

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processor = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
    )



    embedding_model = SentenceTransformer(model_rag_path, trust_remote_code=True)
    embedding_model.to(device)

    # instruct_model, instruct_vis_processor, instruct_txt_processor = load_model_and_preprocess(
    #     name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device
    # )

    print('Loading models successfully at ', device)


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
                              next_event=feature.next_event, time_gap=feature.time_gap,
                              semantic_name=semantic_name)
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
    semantic_name = feature.semantic_name
    # semantic_name = feature.semantic_name
    # start_hour = feature.start_hour
    # end_hour = feature.end_hour
    # is_weekend = feature.is_weekend

    # Perform search
    raw_result = retrieve_image(concept_query=query, embed_model=model, txt_processor=txt_processor,
                                semantic_name=semantic_name)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)

    # Automatic run Logging query string
    # automatic_logging(results=results, output_file_name='ntcir_automatic_logging')

    return results


# @router.post(
#     "/question_answering",
#     status_code=status.HTTP_200_OK,
# )
# async def question_answering(feature: FeatureModelSingleSearch, api_key: APIKey = Depends(get_api_key)):
#     # Question answering endpoint, give the question and some filters if possible
#     # Output: dictionary with key: question.
#     query = feature.query
#     topic = feature.topic
#     semantic_name = feature.semantic_name
#     start_hour = feature.start_hour
#     end_hour = feature.end_hour
#     is_weekend = feature.is_weekend

#     answer = process_result(query=query, blip2_embed_model=model, blip2_txt_processor=txt_processor,
#                             instruct_model=instruct_model, instruct_vis_processor=instruct_vis_processor,
#                             device=device, semantic_name=semantic_name, start_hour=start_hour,
#                             end_hour=end_hour, is_weekend=is_weekend)

#     return {"answer": answer}

@router.post(
    "/conversational_search",
    status_code=status.HTTP_200_OK,
)
async def conversation_search(feature: FeatureModelConversationalSearch, api_key: APIKey = Depends(get_api_key)):
    # Chat to retrieve images
    # Input: query, previous chat of users
    # Output: the list of results and textual answer
    query = feature.query
    previous_chat = feature.previous_chat
    if '?' in query:
        # perform RAG
        result, return_answer = rag_question_answering(query=query, previous_chat=previous_chat,
                                                       embedding_model=embedding_model)
    else:
        result, return_answer = chat(query=query, previous_chat=previous_chat, model=model,
                                     txt_processors=txt_processor)
    output_dict = {'results': result, 'textual_answer': return_answer}
    return output_dict


@router.post(
    "/visual_similarity",
    status_code=status.HTTP_200_OK,
)
async def visual_similarity(feature: FeatureModelVisualSimilarity, api_key: APIKey = Depends(get_api_key)):
    # Relevance feedback with embedding similarity search
    # Input: query and filters for search and filters, image_id for embedding similarity.
    # Output: list of dict with key current_event.
    query = feature.query
    image_id = feature.image_id
    if query == '' and len(image_id) == 0:
        # return ramdom 100 images id
        col = ["ImageID", "new_name", 'event_id', 'local_time', 'day_of_week', 'similar_image']
        query_template = {
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "functions": [{"random_score": {}}]
                }
            },
            "_source": col, "size": 100}
        query_template = json.dumps(query_template)
        raw_result = send_request_to_elasticsearch(HOST, INDICES, query_template)
    elif query != '' and len(image_id) == 0:
        # retrieve by query
        raw_result = retrieve_image(concept_query=query, embed_model=model, txt_processor=txt_processor)
    else:
        # Calculate the mean embedding of all image input
        mean_embedding = calculate_mean_emb(image_id=image_id)
        # Perform search by image embedding
        raw_result = relevance_image_similar(image_embedding=mean_embedding, query=query, image_id=image_id)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]

    results = add_image_link(results)
    return results


def include_router(app):
    app.include_router(router)
    app.add_middleware(RequestTimestampMiddleware, router_path='/predict')


initialize_resources()
