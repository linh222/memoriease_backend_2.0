import os

import openai
from dotenv import load_dotenv

from app.config import root_path
from app.predictions.predict import retrieve_image

load_dotenv(str(root_path) + '/.env')
openai.api_key = os.environ.get("openai_key")


def query_expansion(concept_query: str, embed_model, txt_processor):
    # Create multiple queries, retrieve the results and get the agg results

    # Create paraphrase query from original query
    paraphrased_query = chatgpt_query_paraphrase(concept_query)

    # Fetch original and paraphrased results in a single call
    original_result = retrieve_image(concept_query, embed_model, txt_processor)
    paraphrase_result = retrieve_image(paraphrased_query, embed_model, txt_processor)

    original_result = original_result['hits']['hits']
    paraphrase_result = paraphrase_result['hits']['hits']

    # Use dictionary comprehensions to create original_dict and paraphrase_dict
    original_dict = {retrieved_result['_id']: retrieved_result['_score'] for retrieved_result in original_result}
    paraphrase_dict = {retrieved_result['_id']: retrieved_result['_score'] for retrieved_result in paraphrase_result}

    merge_dict = {}

    for image_id in original_dict.keys():
        # Avoid try-except block and use get() to handle missing keys
        original_score = original_dict.get(image_id, 0)
        paraphrase_score = paraphrase_dict.get(image_id, 0)
        merge_score = original_score * 0.7 + paraphrase_score * 0.3
        merge_dict[image_id] = merge_score

    # Use list comprehension to create final_list
    final_list = [{'_id': image_id, '_score': score, '_source': get_source(image_id, original_result)}
                  for image_id, score in merge_dict.items()]

    return final_list


def get_source(_id, data_dict):
    for item in data_dict:
        if item['_id'] == _id:
            return item['_source']
    return {}


def chatgpt_query_paraphrase(query):
    prompt = 'Create a new sentence with different words with the same meaning of this sentence: '
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt + query},
        ]
    )
    return response['choices'][0]['message']['content']

# if __name__ == "__main__":
#     # # Remote server
#     import time
#
#     start_time = time.time()
#     device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                       model_type="coco", is_eval=True,
#                                                                       device=device)
#     print("cuda" if torch.cuda.is_available() else "cpu")
#     print(time.time() - start_time, 'seconds')
#
#     result = query_expansion(concept_query="""Exotic birds. Find examples of multicoloured parrots (real or fake) in a
#     tree at our rented house in Thailand.""", embed_model=model, txt_processor=txt_processors)
#
#     with open('{}/app/evaluation_model/result.json'.format(root_path), 'w') as f:
#         json.dump(result, f)
