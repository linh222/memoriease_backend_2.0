from fastapi import APIRouter, status, Depends
from fastapi.openapi.models import APIKey

from app.api_key import get_api_key
from app.apis.api_utils import add_image_link
from app.predictions.relevance_feedback import relevance_image_similar, calculate_mean_emb
from .schemas import FeatureModelRelevanceSearch

router = APIRouter()


@router.post(
    "/relevance_feedback",
    status_code=status.HTTP_200_OK,
)
async def relevance_feedback(feature: FeatureModelRelevanceSearch, api_key: APIKey = Depends(get_api_key)):
    query = feature.query
    image_id = feature.image_id
    semantic_name = feature.semantic_name
    mean_embedding = calculate_mean_emb(image_id=image_id)
    raw_result = relevance_image_similar(image_embedding=mean_embedding, query=query, semantic_name=semantic_name)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)
    return results


# @router.post(
#     "/predict_pseudo_relevance_feedback",
#     status_code=status.HTTP_200_OK,
# )
# async def predict_image_peuso_rf(feature: FeatureModelSingleNTCIRSearch, api_key: APIKey = Depends(get_api_key)):
#     query = feature.query
#     semantic_name = feature.semantic_name
#
#     raw_result = pseudo_relevance_feedback(concept_query=query, embed_model=model, txt_processor=txt_processor,
#                                            semantic_name=semantic_name, start_hour=feature.start_hour,
#                                            end_hour=feature.end_hour, is_weekend=feature.is_weekend, top_k=10)
#     results = [{'current_event': result} for result in raw_result['hits']['hits']]
#     results = add_image_link(results)
#
#     # Automatic run Logging query string
#     automatic_logging(results=results, output_file_name='ntcir_automatic_logging')
#
#     return results


def include_router(app):
    app.include_router(router)
