import json
import os
import pickle

import openai
import requests
from dotenv import load_dotenv

from app.apis.api_utils import add_image_link
from app.config import HOST, INDICES
from app.config import root_path
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.temporal_predict import time_processing_event, send_request_by_event
from app.predictions.utils import build_query_template

load_dotenv(str(root_path) + '/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

predefined_time_period = ['early morning', 'morning', 'afternoon', 'night']
predefined_day_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
predefined_city = pickle.load(open('{}/app/predictions/mysceal_nlp_utils/common/city.pkl'.format(root_path), 'rb'))
predefined_city = ' '.join(predefined_city).split(' ')
predefined_city = list(set(predefined_city))
predefined_city.remove('')


def send_gpt_request(query):
    prompt = """Query: Having lunch with Dermot, who was a guest speaker at my lecture. After lunch, he gave a 
    lecture to my class about Lessons in Innovation & Entrepreneurship while I was sitting in the front row. It was 
    in November 2019. Result: [{'term': {'concept': 'Having lunch with Dermot, who was a guest speaker at my 
    lecture'}},{'term': {'after_concept': 'After lunch, he gave a lecture to my class about Lessons in Innovation & 
    Entrepreneurship in dublin city university while I was sitting in the front row.'}},{'term': {'before_concept': 
    ''}},{'term': {'time_period': ''}}, {'term': {'day_of_week': ''}}, {'range': {'local_time': {'gte': '2019-11-01', 
    'lte': '2019-11-30'}}}, {'term': {'city': 'dublin'}}] Query: Drinks on top of the Bangkok. Taking a drink on a 
    rooftop bar at night in Bangkok. It was on the same day that I flew into Bangkok. In 2019 in September. Result: [
    {'term': {'concept': 'Drinks on top of the Bangkok. Taking a drink on a rooftop bar at night in Bangkok.'}}, 
    {'term': {'after_concept': ''}}, {'term': {'before_concept': 'It was on the same day that I flew into Bangkok'}}, 
    {'term': {'time_period': 'night'}}, {'term': {'day_of_week': ''}}, {'range': {'local_time': {'gte': '2019-09-01', 
    'lte': '2019-09-30'}}}, {'term': {'city': 'bangkok'}}] Query: I remember that there was a man in a pink t-shirt 
    in front of a wall looking at the water. Result: [{'term': {'concept': 'there was a man in a pink t-shirt in 
    front of a wall looking at the water.'}}, {'term': {'after_concept': ''}}, {'term': {'before_concept': ''}}, 
    {'term': {'time_period': ''}}, {'term': {'day_of_week': ''}}, {'range': {'local_time': {'gte': '2019-01-01', 
    'lte': '2020-06-30'}}}, {'term': {'city': ''}}] Query: I was getting an eye test after looking for a new pair of 
    glasses. I remember the optician had a red sweater with a reindeer on it and that I had to look into some 
    machines. After the eye test I went shopping for groceries. It was a few days before Christmas. Result: [{'term': 
    {'concept': 'I was getting an eye test after looking for a new pair of glasses. I remember the optician had a red 
    sweater with a reindeer on it and that I had to look into some machines'}}, {'term': {'after_concept': 'Shopping 
    for groceries'}}, {'term': {'before_concept': ''}}, {'term': {'time_period': ''}}, {'term': {'day_of_week': ''}}, 
    {'range': {'local_time': {'gte': '2019-12-01', 'lte': '2019-12-24'}}}, {'term': {'city': ''}}]"""
    query_format = f"Query: {query}\n Result: "
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct-0914",
        prompt=prompt + query_format,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def construct_filter(query):
    # Input: The query
    # Output: The elasticsearch query template with KNN retrieval and filters
    main_event_context, previous_event_context, after_event_context = '', '', ''
    filters = eval(send_gpt_request(query))
    new_filters = []
    for each_filter in filters:
        if 'concept' in each_filter[list(each_filter.keys())[0]]:
            # perform convert embedding
            main_event_context = each_filter['term']['concept']
        elif 'after_concept' in each_filter[list(each_filter.keys())[0]]:
            if each_filter['term']['after_concept'] != '':
                # perform temporal search
                after_event_context = each_filter['term']['after_concept']
        elif 'before_concept' in each_filter[list(each_filter.keys())[0]]:
            if each_filter[list(each_filter.keys())[0]]['before_concept'] != '':
                # perform temporal search
                previous_event_context = each_filter['term']['before_concept']
        else:
            new_filters.append(filter)
    return new_filters, main_event_context, previous_event_context, after_event_context


def retrieve_result(main_event_context: str, previous_event_context: str, after_event_context: str, filters: list,
                    embed_model, txt_processor, size=100):
    if main_event_context == '':
        ValueError("The query should not be blank.")

    # examine the filters
    new_filters = []
    for each_filter in filters:
        if 'time_period' in each_filter[list(each_filter.keys())[0]]:
            if each_filter['term']['time_period'] in predefined_time_period and \
                    each_filter['term']['time_period'] != '':
                new_filters.append(each_filter)
        elif 'day_of_week' in each_filter[list(each_filter.keys())[0]]:
            if each_filter['term']['day_of_week'] in predefined_day_of_week and \
                    each_filter['term']['day_of_week'] != '':
                new_filters.append(each_filter)
        elif 'city' in each_filter[list(each_filter.keys())[0]]:
            if each_filter['term']['city'] in predefined_city and each_filter['term']['city'] != '':
                new_filters.append(each_filter)
        else:
            new_filters.append(each_filter)

    # Construct body request
    text_embedding = extract_query_blip_embedding(main_event_context, embed_model, txt_processor)
    query_template = build_query_template(filter=new_filters, text_embedding=text_embedding, size=size)
    query_template = json.dumps(query_template)
    url = f"{HOST}/{INDICES}/_search"

    # send request to elastic search
    with requests.Session() as session:
        try:
            response = session.post(url, data=query_template, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            results = response.json()
        except requests.exceptions.RequestException as e:
            ValueError(e)
            return None

    # check if null value
    if 'hits' not in results:
        ValueError("Modify the query to get meaningful results")

    results = [{'current_event': each_result} for each_result in results['hits']['hits']]
    # add image link
    results = add_image_link(results)

    # perform temporal search
    if previous_event_context != '' or after_event_context != '':
        new_results = []
        previous_text_embedding, after_text_embedding = '', ''
        list_previous_event_query, list_next_event_query = [], []
        if len(results) > 0:
            if previous_event_context != '':
                previous_text_embedding = extract_query_blip_embedding(previous_event_context,
                                                                       embed_model, txt_processor)
            if after_event_context != '':
                after_text_embedding = extract_query_blip_embedding(after_event_context, embed_model, txt_processor)
            for event in results:
                new_results.append(event)

                if previous_event_context != '':
                    list_previous_event_query = time_processing_event(list_event_query=list_previous_event_query,
                                                                      main_event=event, time_gap=2,
                                                                      time_period='', location='', keyword='',
                                                                      weekday='', embed=previous_text_embedding,
                                                                      type='previous')
                if after_event_context != '':
                    list_next_event_query = time_processing_event(list_event_query=list_next_event_query,
                                                                  main_event=event, time_gap=2,
                                                                  time_period='', location='', keyword='',
                                                                  weekday='', embed=after_text_embedding,
                                                                  type='next')
        if previous_event_context != '':
            new_results = send_request_by_event(new_result=new_results, event_query=list_previous_event_query,
                                                type='previous')
        if after_event_context != '':
            new_results = send_request_by_event(new_result=new_results, event_query=list_next_event_query,
                                                type='next')
        for index in range(len(new_results)):
            overall_score = new_results[index]['current_event']['_score'] * 0.6
            if "previous_event" in new_results[index].keys() and new_results[index]['previous_event']['_id'] is not None:
                overall_score = overall_score + (new_results[index]['previous_event']['_score'] * 0.2)
            if "next_event" in new_results[index].keys() and new_results[index]['next_event']['_id'] is not None:
                overall_score = overall_score + (new_results[index]['next_event']['_score'] * 0.2)
            new_results[index]['overall_score'] = overall_score
        new_results = sorted(new_results, key=lambda d: d['overall_score'], reverse=True)
        return new_results
    else:
        return results

#
# if __name__ == "__main__":
#     import time
#
#     import torch
#     from LAVIS.lavis.models import load_model_and_preprocess
#     from app.config import root_path
#
#     start_time = time.time()
#     device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                       model_type="coco", is_eval=True,
#                                                                       device=device)
#     print("cuda" if torch.cuda.is_available() else "cpu")
#     print(time.time() - start_time, 'seconds')
#     filter, main_event, previous_event, after_event = construct_filter('I am at home in the afternoon of 20th '
#                                                                        'January 2019 in Dublin')
#
#     result = retrieve_result(main_event_context=main_event, previous_event_context=previous_event,
#                              after_event_context=after_event,
#                              filters=filter, embed_model=1, txt_processor=1, size=100)
#     print(result)
