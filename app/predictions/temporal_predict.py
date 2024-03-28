import json
from datetime import datetime, timedelta

import requests

from app.apis.api_utils import extract_date_imagename
from app.config import HOST, INDICES, IMAGE_SERVER, IMAGE_EXT
from app.predictions.blip_extractor import extract_query_blip_embedding
from app.predictions.predict import retrieve_image
from app.predictions.utils import process_query, build_query_template, construct_filter, calculate_overall_score


def time_processing_event(list_event_query, main_event, time_gap, time_period, location, keyword, weekday, embed,
                          temporal_type) -> list:
    # Process the time from the main event. If type is previous event then minus 1 minutes, else plus 1 minutes
    # And create query template to send elastic search
    if temporal_type == 'previous':
        main_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) - timedelta(hours=0, minutes=1)
        gap_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) - timedelta(hours=time_gap, minutes=0)
        time = [str(gap_event_time).replace(' ', 'T'), str(main_event_time).replace(' ', 'T')]
    elif temporal_type == 'next':
        main_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) + timedelta(hours=0, minutes=1)
        gap_event_time = datetime.fromisoformat(
            main_event['current_event']['_source']['local_time']) + timedelta(hours=time_gap, minutes=0)
        time = [str(main_event_time).replace(' ', 'T'), str(gap_event_time).replace(' ', 'T')]
    else:
        ValueError('Type is wrong')
        time = None
    query_dict = {'time_period': time_period, 'location': location,
                  'list_keyword': keyword,
                  'weekday': weekday,
                  'time_filter': time}
    previous_filter = construct_filter(query_dict)
    previous_filter_template = build_query_template(previous_filter, embed, size=1)
    list_event_query.append(previous_filter_template)
    return list_event_query


def send_request_by_event(main_event_result, event_query, temporal_type):
    # Send request for list of temporal event
    payload = ''
    for query in event_query:
        payload += json.dumps({'index': INDICES}) + '\n'
        payload += json.dumps(query) + '\n'
    # Send multi search request to elastic search
    event_response = requests.get(HOST + '/_msearch?pretty',
                                  headers={'Content-Type': 'application/json'}, data=payload)
    result = event_response.json()

    for index in range(len(main_event_result)):
        # Add image link to results
        if len(result['responses'][index]['hits']['hits']) == 1:
            image_id_result = result['responses'][index]['hits']['hits'][0]
            image_id = image_id_result['_source']['ImageID']
            image_name, year_month, day = extract_date_imagename(image_id)
            result['responses'][index]['hits']['hits'][0]['_source'][
                'image_link'] = '{}/{}/{}/{}.{}'.format(IMAGE_SERVER, year_month, day, image_name, IMAGE_EXT)
            if temporal_type == 'previous':
                main_event_result[index]['previous_event'] = result['responses'][index]['hits']['hits'][0]
            elif temporal_type == 'next':
                main_event_result[index]['next_event'] = result['responses'][index]['hits']['hits'][0]
            else:
                ValueError('type is wrong')
        else:
            if temporal_type == 'previous':
                main_event_result[index]['previous_event'] = {'_id': None}
            elif temporal_type == 'next':
                main_event_result[index]['next_event'] = {'_id': None}
            else:
                ValueError('type is wrong')

    return main_event_result


def temporal_search(concept_query, embed_model, txt_processor,
                    previous_event='', next_event='', time_gap=1, semantic_name=None):
    # Search for main event, previous event and next event
    # Input: query for three type of temporal, filters, timegap
    # Output: Results list of dicts with three keys: current event, previous event, after event
    main_event_result = retrieve_image(concept_query=concept_query, embed_model=embed_model,
                                       txt_processor=txt_processor, semantic_name=semantic_name)
    main_event_result = main_event_result['hits']['hits']
    full_result = []
    list_next_event_query, list_previous_event_query = [], []
    previous_processed_query, previous_list_keyword, previous_time_period, previous_weekday, previous_time_filter, \
        previous_text_embedding, previous_location = '', '', '', '', ['', ''], '', ''
    next_processed_query, next_list_keyword, next_time_period, next_weekday, next_time_filter, \
        next_text_embedding, next_location = '', '', '', '', ['', ''], '', ''

    if len(main_event_result) > 0:
        if previous_event != "":
            # Construct filters and text embedding for previous event
            previous_processed_query, previous_list_keyword, previous_time_period, previous_weekday, \
                previous_time_filter, previous_location = process_query(previous_event)
            previous_text_embedding = extract_query_blip_embedding(query=previous_processed_query, model=embed_model,
                                                                   processor=txt_processor)
        if next_event != "":
            # Construct filters and text embedding for next event
            next_processed_query, next_list_keyword, next_time_period, next_weekday, \
                next_time_filter, next_location = process_query(next_event)
            next_text_embedding = extract_query_blip_embedding(query=next_processed_query, model=embed_model,
                                                               processor=txt_processor)
        for main_event in main_event_result:
            current_result = main_event
            main_event = {'current_event': current_result}
            full_result.append(main_event)
            # Process query previous event
            if previous_event != "":
                list_previous_event_query = time_processing_event(list_event_query=list_previous_event_query,
                                                                  main_event=main_event, time_gap=time_gap,
                                                                  time_period=previous_time_period,
                                                                  location=previous_location,
                                                                  keyword=previous_list_keyword,
                                                                  weekday=previous_weekday,
                                                                  embed=previous_text_embedding,
                                                                  temporal_type='previous')
            # Process query previous event
            if next_event != "":
                list_next_event_query = time_processing_event(list_event_query=list_next_event_query,
                                                              main_event=main_event, time_gap=time_gap,
                                                              time_period=next_time_period,
                                                              location=next_location,
                                                              keyword=next_list_keyword,
                                                              weekday=next_weekday,
                                                              embed=next_text_embedding,
                                                              temporal_type='next')
    if previous_event != "":
        full_result = send_request_by_event(main_event_result=full_result, event_query=list_previous_event_query,
                                            temporal_type='previous')

    if next_event != "":
        full_result = send_request_by_event(main_event_result=full_result, event_query=list_next_event_query,
                                            temporal_type='next')

    full_result = calculate_overall_score(full_result)

    return full_result
