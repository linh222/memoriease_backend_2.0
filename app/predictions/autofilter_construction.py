import json
import os
import pickle

import openai
from openai import OpenAI
import requests
from dotenv import load_dotenv

from app.apis.api_utils import add_image_link
from app.config import HOST, INDICES
from app.config import root_path
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.temporal_predict import time_processing_event, send_request_by_event
from app.predictions.utils import build_query_template, send_request_to_elasticsearch, calculate_overall_score

load_dotenv(str(root_path) + '/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

predefined_time_period = ['early morning', 'morning', 'afternoon', 'night']
predefined_day_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
predefined_city = pickle.load(open('{}/app/predictions/mysceal_nlp_utils/common/city.pkl'.format(root_path), 'rb'))
predefined_city = ' '.join(predefined_city).split(' ')
predefined_city = list(set(predefined_city))
predefined_city.remove('')


def send_gpt_request(query):
    # Ask chatgpt to complete the text
    # Input: query, plus with prompt and instruction format for chatgpt
    # Output: the result
    # TODO Prompt engineering to improve the accuracy
    # TODO Iterate until chatgpt give the correct format

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful query extractor to extract relevant information from a query to form a "
                           "json file to query in elastic search. The set of valid values for time_perios is ["
                           "'morning', 'afternoon', 'night', 'late night', 'early morning'] and for day_of_week is ["
                           "'monday', 'tuesday', 'webnesday', 'thursday', 'friday', 'saturday', 'sunday'] and city is "
                           "only the name of the city\nHere are some examples of the task:\nPrompt examples for "
                           "filter construction.\nExtract the information from query to the result as follow: "
                           "\nQuery: Having lunch with Dermot, who was a guest speaker at my lecture. After lunch, "
                           "he gave a lecture to my class about Lessons in Innovation & Entrepreneurship while I was "
                           "sitting in the front row. It was in November 2019.\nResult: [{'term': {'concept': 'Having "
                           "lunch with Dermot, who was a guest speaker at my lecture'}}, {'term': {'after_concept': "
                           "'After lunch, he gave a lecture to my class about Lessons in Innovation & "
                           "Entrepreneurship in Dublin City University while I was sitting in the front row.'}},"
                           "{'term': {'before_concept':''}},{'term': {'time_period': 'afternoon'}}, {'term': {"
                           "'day_of_week': ''}}, {'range': {'local_time': {'gte': '2019-11-01',"
                           "\n'lte': '2019-11-30'}}}, {'term': {'city': 'dublin'}}]\nQuery: When did I buy that model "
                           "train? I remember it was a marklin brand train and I bought it at the weekend. Jer "
                           "convinced me to buy it when having coffee and I bought it immediately after coffee. It "
                           "was in June 2019.\nResult: [{'term': {'concept': 'When did I buy that model train? I "
                           "remember it was a marklin brand train'}}, {'term': {'after_concept': ''}}, "
                           "{'term': {'before_concept': 'Jer convinced me to buy it when having coffee'}}, "
                           "{'term': {'time_period': ''}}, {'term': {'day_of_week': ''}}, {'range': {'local_time': {"
                           "'gte': '2019-06-01', 'lte': '2019-06-30'}}}, {'term': {'city': ''}}]\nQuery: I remember a "
                           "man in a blue coat walking a dog in the countryside in Ireland on a sunny afternoon in "
                           "December on Christmas Day.\nResult: [{'term': {'concept': 'a man in a blue coat walking a "
                           "dog in the countryside on a sunny afternoon'}}, {'term': {'after_concept': ''}},"
                           "\n{'term': {'before_concept': ''}}, {'term': {'time_period': 'afternoon'}}, "
                           "{'term': {'day_of_week': ''}},\n{'range': {'local_time': {'gte': '2019-12-25', "
                           "'lte': '2019-12-25'}}}, {'term': {'city': 'Ireland'}}]\nQuery: I remember that there was "
                           "a man in a pink t-shirt in front of a wall looking at the water.\nResult: [{'term': {"
                           "'concept': 'there was a man in a pink t-shirt in front of a wall looking at the "
                           "water.'}},\n{'term': {'after_concept': ''}}, {'term': {'before_concept': ''}}, "
                           "{'term': {'time_period': ''}}, {'term': {'day_of_week': ''}},\n{'range': {'local_time': {"
                           "'gte': '2019-01-01', 'lte': '2020-06-30'}}}, {'term': {'city': ''}}]\nQuery: Damn it, "
                           "my car has a flat tyre. What was the name of the car service/repair company that I used "
                           "in the summer of 2019? I want to call them to get my car fixed.\nResult: [{'term': {"
                           "'concept': 'Damn it, my car has a flat tyre. The name of the car service/repair company "
                           "that I used to get my car fixed.'}},\n{'term': {'after_concept': ''}}, {'term': {"
                           "'before_concept': ''}}, {'term': {'time_period': 'morning'}}, {'term': {'day_of_week': "
                           "''}},\n{'range': {'local_time': {'gte': '2019-04-01', 'lte': '2019-08-31'}}}, "
                           "{'term': {'city': ''}}]\nQuery: I was getting an eye test after looking for a new pair of "
                           "glasses. I remember the optician had a red sweater with a reindeer on it and that I had "
                           "to look into some machines. After the eye test, I went shopping for groceries. It was a "
                           "few days before Christmas.\nResult: [{'term': {'concept': 'I was getting an eye test. I "
                           "remember the optician had a red sweater with a reindeer on it and that I had to look into "
                           "some machines'}}, {'term': {'after_concept': 'Shopping for groceries'}}, "
                           "{'term': {'before_concept': 'looking for a new pair of glasses'}}, {'term': {"
                           "'time_period': ''}}, {'term': {'day_of_week': ''}},\n{'range': {'local_time': {'gte': "
                           "'2019-12-15', 'lte': '2019-12-24'}}}, {'term': {'city': ''}}]"
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def construct_filter(query):
    # Construct query from textual query.
    # Input: The query
    # Output: The elasticsearch query template with KNN retrieval and filters
    main_event_context, previous_event_context, after_event_context = '', '', ''
    filters = eval(send_gpt_request(query)) # Convert to dictionary
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
            # Filters with semantic name, hour, city, ...
            new_filters.append(each_filter)
    return new_filters, main_event_context, previous_event_context, after_event_context


def retrieve_result(main_event_context: str, previous_event_context: str, after_event_context: str, filters: list,
                    embed_model, txt_processor, size=100):
    # Retrieve the data from the constructed filters
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
    results = send_request_to_elasticsearch(HOST, INDICES, query_template)

    # check if null value
    if results is None or 'hits' not in results:
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
                                                                      temporal_type='previous')
                if after_event_context != '':
                    list_next_event_query = time_processing_event(list_event_query=list_next_event_query,
                                                                  main_event=event, time_gap=2,
                                                                  time_period='', location='', keyword='',
                                                                  weekday='', embed=after_text_embedding,
                                                                  temporal_type='next')
        if previous_event_context != '':
            new_results = send_request_by_event(main_event_result=new_results, event_query=list_previous_event_query,
                                                temporal_type='previous')
        if after_event_context != '':
            new_results = send_request_by_event(main_event_result=new_results, event_query=list_next_event_query,
                                                temporal_type='next')

        new_results = calculate_overall_score(new_results)
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
