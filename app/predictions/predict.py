import json

from app.config import HOST, INDICES
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.utils import process_query, construct_filter, build_query_template, send_request_to_elasticsearch,\
    extract_advanced_filter, add_advanced_filters


def retrieve_image(concept_query: str, embed_model, txt_processor, semantic_name='', size=100):
    # Advanced search with tag
    returned_query, advanced_filters = extract_advanced_filter(concept_query)
    # Processing the query
    processed_query, list_keyword, time_period, weekday, time_filter, location = process_query(returned_query)
    text_embedding = extract_query_blip_embedding(processed_query, embed_model, txt_processor)

    query_dict = {
        "time_period": time_period,
        "location": location,
        "list_keyword": list_keyword,
        "weekday": weekday,
        "time_filter": time_filter,
        "semantic_name": semantic_name
    }
    if len(advanced_filters) > 0:
        query_dict = add_advanced_filters(advanced_filters, query_dict)
    filters = construct_filter(query_dict)
    query_template = build_query_template(filters, text_embedding, size=size)
    query_template = json.dumps(query_template)

    results = send_request_to_elasticsearch(HOST, INDICES, query_template)
    return results

# if __name__ == "__main__":
#     # Remote server
#     import time
#     import torch
#     from LAVIS.lavis.models import load_model_and_preprocess
#     from app.config import root_path
#     start_time = time.time()
#     device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                       model_type="coco", is_eval=True,
#                                                                       device=device)
#     print("cuda" if torch.cuda.is_available() else "cpu")
#     print(time.time() - start_time, 'seconds')
#     result = retrieve_image(concept_query="""I go homewares shopping  in Ireland in 2019""",
#                             embed_model=model, txt_processor=txt_processors, semantic_name="", start_hour=20,
#                             end_hour=24, is_weekend=0, size=100)
#     with open('{}/app/evaluation_model/result.json'.format(root_path), 'w') as f:
#         json.dump(result, f)
