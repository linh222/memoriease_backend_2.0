import json
from datetime import datetime, timedelta

import requests

from app.config import HOST, INDICES, IMAGE_SERVER
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.predict import retrieve_image
from app.predictions.utils import process_query, build_query_template, construct_filter


def time_processing_event(list_event_query, main_event, time_gap, time_period, location, keyword, weekday, embed,
                          type) -> list:
    # list_event_query = []

    if type == 'previous':
        main_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) - timedelta(hours=0,
                                                                              minutes=1)
        gap_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) - timedelta(
            hours=time_gap,
            minutes=0)
        time = [str(gap_event_time).replace(' ', 'T'), str(main_event_time).replace(' ', 'T')]
    elif type == 'next':
        main_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) + timedelta(hours=0,
                                                                              minutes=1)
        gap_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) + timedelta(
            hours=time_gap,
            minutes=0)
        time = [str(main_event_time).replace(' ', 'T'), str(gap_event_time).replace(' ', 'T')]
    else:
        ValueError('Type is wrong')
        time = None
    query_dict = {'time_period': time_period, 'location': location,
                  'list_keyword': keyword,
                  'weekday': weekday,
                  'time_filter': time}
    previous_filter, previous_must = construct_filter(query_dict)
    previous_filter_template = build_query_template(previous_filter, previous_must, embed, size=1)
    list_event_query.append(previous_filter_template)
    return list_event_query


def send_request_by_event(new_result, event_query, type):
    payload = ''
    for query in event_query:
        payload += json.dumps({'index': INDICES}) + '\n'
        payload += json.dumps(query) + '\n'

    event_response = requests.get(HOST + '/_msearch?pretty',
                                  headers={'Content-Type': 'application/json'}, data=payload)
    result = event_response.json()

    for index in range(len(new_result)):
        if len(result['responses'][index]['hits']['hits']) == 1:
            image_id_result = result['responses'][index]['hits']['hits'][0]
            image_id = image_id_result['_source']['ImageID']
            year_month = image_id[:6]
            day = image_id[6:8]
            image_name = image_id[0:-4]
            result['responses'][index]['hits']['hits'][0]['_source'][
                'image_link'] = '{}/{}/{}/{}.webp'.format(IMAGE_SERVER,
                                                          year_month, day, image_name)
            if type == 'previous':
                new_result[index]['previous_event'] = result['responses'][index]['hits']['hits'][0]
            elif type == 'next':
                new_result[index]['next_event'] = result['responses'][index]['hits']['hits'][0]
            else:
                ValueError('type is wrong')
        else:
            if type == 'previous':
                new_result[index]['previous_event'] = {'_id': None}
            elif type == 'next':
                new_result[index]['next_event'] = {'_id': None}
            else:
                ValueError('type is wrong')

    return new_result


def temporal_search(concept_query, embed_model, txt_processor,
                    previous_event='', next_event='', time_gap=1, semantic_name=None):
    main_event_result = retrieve_image(concept_query=concept_query, embed_model=embed_model,
                                       txt_processor=txt_processor, semantic_name=semantic_name, )
    main_event_result = main_event_result['hits']['hits']
    new_result = []
    list_next_event_query, list_previous_event_query = [], []
    previous_processed_query, previous_list_keyword, previous_time_period, previous_weekday, previous_time_filter, \
        previous_text_embedding, previous_location = '', '', '', '', ['', ''], '', ''
    next_processed_query, next_list_keyword, next_time_period, next_weekday, next_time_filter, \
        next_text_embedding, next_location = '', '', '', '', ['', ''], '', ''

    if len(main_event_result) > 0:
        if previous_event != "":
            previous_processed_query, previous_list_keyword, previous_time_period, previous_weekday, \
                previous_time_filter, previous_location = process_query(previous_event)
            previous_text_embedding = extract_query_blip_embedding(query=previous_processed_query, model=embed_model,
                                                                   processor=txt_processor)
        if next_event != "":
            next_processed_query, next_list_keyword, next_time_period, next_weekday, \
                next_time_filter, next_location = process_query(next_event)
            next_text_embedding = extract_query_blip_embedding(query=next_processed_query, model=embed_model,
                                                               processor=txt_processor)
        for main_event in main_event_result:
            current_result = main_event
            main_event = {'current_event': current_result}
            new_result.append(main_event)
            # Process query previous event
            if previous_event != "":
                list_previous_event_query = time_processing_event(list_event_query=list_previous_event_query,
                                                                  main_event=main_event, time_gap=time_gap,
                                                                  time_period=previous_time_period,
                                                                  location=previous_location,
                                                                  keyword=previous_list_keyword,
                                                                  weekday=previous_weekday,
                                                                  embed=previous_text_embedding, type='previous')
            if next_event != "":
                list_next_event_query = time_processing_event(list_event_query=list_next_event_query,
                                                              main_event=main_event, time_gap=time_gap,
                                                              time_period=next_time_period,
                                                              location=next_location,
                                                              keyword=next_list_keyword,
                                                              weekday=next_weekday,
                                                              embed=next_text_embedding, type='next')
    if previous_event != "":
        new_result = send_request_by_event(new_result=new_result, event_query=list_previous_event_query,
                                           type='previous')

    if next_event != "":
        new_result = send_request_by_event(new_result=new_result, event_query=list_next_event_query,
                                           type='next')

    for index in range(len(new_result)):
        overall_score = new_result[index]['current_event']['_score'] * 0.6
        if "previous_event" in new_result[index].keys() and new_result[index]['previous_event']['_id'] is not None:
            overall_score = overall_score + (new_result[index]['previous_event']['_score'] * 0.2)
        if "next_event" in new_result[index].keys() and new_result[index]['next_event']['_id'] is not None:
            overall_score = overall_score + (new_result[index]['next_event']['_score'] * 0.2)
        new_result[index]['overall_score'] = overall_score
    new_result = sorted(new_result, key=lambda d: d['overall_score'], reverse=True)

    return new_result
