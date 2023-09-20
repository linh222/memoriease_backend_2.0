import pickle
import re

from app.config import root_path
from app.predictions.mysceal_nlp_utils.common import locations
from app.predictions.mysceal_nlp_utils.pos_tag import Tagger


def time_contructor(date):
    start_time1, end_time1 = '', ''
    if '2019' in date:
        date = date.replace('2019', '')
        year = '2019'
    elif '2020' in date:
        date = date.replace('2020', '')
        year = '2020'
    else:
        year = ''

    list_month = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05',
                  'june': '06', 'july': '07', 'august': '08', 'september': '09', 'october': '10',
                  'november': '11', 'december': '12'}
    month = ''
    for month_query in list_month.keys():
        if month_query in date:
            date = date.replace(month_query, '')
            month = month_query

    day = ''
    numbers_only = re.findall('\d+', date)
    if len(numbers_only) > 0:
        day = numbers_only[0]
    month_31 = ['january', 'march', 'may', 'july', 'august', 'october', 'december']
    month_30 = ['april', 'june', 'september', 'november']
    if year == "" and month == '' and day == '':
        start_time = '2019-01-01'
        end_time = '2020-06-30'
    elif month == '' and year != '':
        start_time = year + '-01-01'
        end_time = year + '-12-31'
    elif month == '' and year != '':
        start_time = year + '-01-01'
        end_time = year + '-12-31'
    elif year == '' and month != '':
        if month not in ['january', 'february', 'march', 'april', 'may', 'june']:
            if month in month_30:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-30'
            elif month in month_31:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-31'
            else:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-28'
        else:
            if month in month_30:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-30'
                start_time1 = '2020-' + list_month[month] + '-01'
                end_time1 = '2020-' + list_month[month] + '-30'
            elif month in month_31:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-31'
                start_time1 = '2020-' + list_month[month] + '-01'
                end_time1 = '2020-' + list_month[month] + '-31'
            else:
                start_time = '2019-' + list_month[month] + '-01'
                end_time = '2019-' + list_month[month] + '-28'
                start_time1 = '2020-' + list_month[month] + '-01'
                end_time1 = '2020-' + list_month[month] + '-28'
    else:
        if month in month_30:
            start_time = year + '-' + list_month[month] + '-01'
            end_time = year + '-' + list_month[month] + '-30'
        elif month in month_31:
            start_time = year + '-' + list_month[month] + '-01'
            end_time = year + '-' + list_month[month] + '-31'
        else:
            start_time = year + '-' + list_month[month] + '-01'
            end_time = year + '-' + list_month[month] + '-28'

    if day != '':
        if len(day) == 1:
            day = '0' + day
        start_time = start_time[:-2] + day
        end_time = end_time[:-2] + day
    if start_time1 != '' and end_time1 != '':
        return [start_time, end_time, start_time1, end_time1]
    else:
        return [start_time, end_time]


valid_time_period = {'early morning': 'early morning', 'dawn': 'early morning', 'sunrise': 'early morning',
                     'daybreak': 'early morning', 'morning': 'morning', 'breakfast': 'morning', 'night': 'night',
                     'nightfall': 'night', 'dusk': 'night', 'dinner': 'night', 'dinnertime': 'night', 'sunset': 'night',
                     'twilight': 'night', 'afternoon': 'afternoon', 'supper': 'afternoon', 'suppertime': 'afternoon',
                     'teatime': 'afternoon', 'late night': 'late night', 'midnight': 'late night', 'evening': 'night'}

valid_location = pickle.load(open('{}/app/predictions/mysceal_nlp_utils/common/city.pkl'.format(root_path), 'rb'))
valid_location = ' '.join(valid_location).split(' ')


def process_query(sent):
    init_tagger = Tagger(locations)
    tags = init_tagger.tag(sent)
    processed_text = ''
    list_keyword = ''
    time_period = ''
    weekday = ''
    location = []
    time_filter = ['', '']

    for index in range(len(tags)):
        if (tags[index][1] == 'NN' or tags[index][1] == 'REGION') and tags[index][0] in valid_location:
            location.append(tags[index][0])
        if tags[index][1] not in ['DATE', 'TIME', 'KEYWORD', 'TIMEPREP', 'TIMEOFDAY', 'WEEKDAY']:
            processed_text = processed_text + ' ' + tags[index][0]
        elif tags[index][1] == 'KEYWORD':
            list_keyword = list_keyword + ', ' + tags[index][0]
        elif tags[index][1] == 'TIMEOFDAY':
            try:
                time_period = valid_time_period[tags[index][0]]
            except:
                time_period = ''
        elif tags[index][1] == 'WEEKDAY':
            weekday = tags[index][0]
        elif tags[index][1] == 'DATE' or (tags[index][1] and tags[index][0] in ['2019', '2020']):
            time_filter = time_contructor(tags[index][0])
        else:
            continue
    list_keyword = list_keyword[2:]
    processed_text = processed_text[1:]
    if len(location) > 0:
        return processed_text, list_keyword, time_period, weekday, time_filter, location[-1]
    else:
        return processed_text, list_keyword, time_period, weekday, time_filter, ''


def construct_filter(query_dict):
    filter = []
    must = []
    if query_dict['time_period'] != '':
        must.append({
            "match": {
                'time_period': {
                    'query': query_dict['time_period']
                }
            }
        })
        filter.append({
            "term": {
                'time_period': query_dict['time_period']
            }
        })
    if query_dict['weekday'] != '':
        must.append({
            "match": {
                'day_of_week': {
                    'query': query_dict['weekday']
                }
            }
        })
        filter.append({
            "term": {
                'day_of_week': query_dict['weekday']
            }
        })
    if query_dict['list_keyword'] != '':
        must.append({
            "match": {
                'Tags': {
                    'query': query_dict['list_keyword']
                }
            }
        })
    if len(query_dict['time_filter']) == 2:
        if query_dict['time_filter'][0] != '' and query_dict['time_filter'][1] != '':
            filter.append({
                "range": {
                    "local_time": {
                        "gte": query_dict['time_filter'][0],
                        "lte": query_dict['time_filter'][1]
                    }
                }
            })
        elif query_dict['time_filter'][0] != '' and query_dict['time_filter'][1] == '':
            filter.append({
                "range": {
                    "local_time": {
                        "gte": query_dict['time_filter'][0]
                    }
                }
            })
        elif query_dict['time_filter'][0] == '' and query_dict['time_filter'][1] != '':
            filter.append({
                "range": {
                    "local_time": {
                        "lte": query_dict['time_filter'][1]
                    }
                }
            })
        else:
            pass
    else:
        filter.append({
            "bool": {
                "should": [
                    {
                        "range": {
                            "local_time": {
                                "gte": query_dict['time_filter'][0],
                                "lte": query_dict['time_filter'][1]
                            }
                        }
                    },
                    {
                        "range": {
                            "local_time": {
                                "gte": query_dict['time_filter'][2],
                                "lte": query_dict['time_filter'][3]
                            }
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        })
    if query_dict['location'] != '':
        must.append({
            'match': {
                'city': {
                    'query': query_dict['location']
                }
            }
        })
        filter.append({
            "term": {
                'city': query_dict['location']
            }
        })
    if 'semantic_name' in query_dict:
        if query_dict['semantic_name'] != '':
            filter.append({
                "term": {
                    'new_name': query_dict['semantic_name']
                }
            })
    if 'start_hour' in query_dict:
        if query_dict['start_hour'] != '':
            filter.append({
                "range": {
                    'hour': {
                        "gte": query_dict['start_hour']
                    }
                }
            })
    if 'end_hour' in query_dict:
        if query_dict['end_hour'] != '':
            filter.append({
                "range": {
                    'hour': {
                        "lte": query_dict['end_hour']
                    }
                }
            })
    if 'is_weekend' in query_dict:
        if query_dict['is_weekend'] != '':
            filter.append({
                "term": {
                    'is_weekend': query_dict['is_weekend']
                }
            })
    if 'groups' in query_dict:
        if query_dict['groups'] != '':
            filter.append({
                "terms": {
                    'group': query_dict['groups']
                }
            })
    return filter, must


def build_query_template(filter, must, text_embedding, size=100):
    col = ["day_of_week", "ImageID", "local_time", "new_name", 'event_id', 'similar_image', 'event', 'group']
    query_template = {

        "knn": {
            "field": "blip_embed",
            "query_vector": text_embedding.tolist(),
            "k": size,
            "num_candidates": 1000,
            # "boost": 1,
            "filter": filter
        },

        "_source": col,
        "size": size,
    }
    return query_template


def automatic_logging(results: list, output_file_name: str):
    logging_data = []
    with open(f'/Users/linhtran/PycharmProject/fastApiProject_blip2/app/evaluation_model/{output_file_name}.csv',
              'r') as file:
        headline = file.readline()
        exist_data = len(file.readlines())
    logging_count = exist_data // 100
    for result in results:
        # score = result['current_event']['_score']
        image_id = result['current_event']['_id']
        text = f'DCU,MEMORIEASE_SAT01,{logging_count},{image_id},0,1'
        logging_data.append(text)

    with open(f'/Users/linhtran/PycharmProject/fastApiProject_blip2/app/evaluation_model/{output_file_name}.csv',
              'a') as file:
        if 'GROUP-ID,RUN-ID,TOPIC-ID,IMAGE-ID,SECONDS-ELAPSED,SCORE' not in headline:
            file.write('GROUP-ID,RUN-ID,TOPIC-ID,IMAGE-ID,SECONDS-ELAPSED,SCORE\n')
        for data in logging_data:
            file.write(data + '\n')
