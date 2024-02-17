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
    # Relevance feedback with embedding similarity search
    # Input: query and filters for search and filters, image_id for embedding similarity.
    # Output: list of dict with key current_event.
    query = feature.query
    image_id = feature.image_id
    # Calculate the mean embedding of all image input
    mean_embedding = calculate_mean_emb(image_id=image_id)
    # Perform search by image embedding
    raw_result = relevance_image_similar(image_embedding=mean_embedding, query=query)
    results = [{'current_event': result} for result in raw_result['hits']['hits']]
    results = add_image_link(results)
    return results


def include_router(app):
    app.include_router(router)
