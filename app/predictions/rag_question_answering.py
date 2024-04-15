import json
import logging
import os

import openai
import requests
from dotenv import load_dotenv
from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer
from openai import OpenAI

from app.config import HOST, RAG_INDICES
from app.config import root_path
from app.predictions.utils import process_query, construct_filter
from app.apis.api_utils import add_image_link

logging.basicConfig(filename='memoriease_backend.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(str(root_path) + '/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


def rag_retriever(question, size):
    url = f"{HOST}/{RAG_INDICES}/_search"

    processed_query, list_keyword, time_period, weekday, time_filter, location = process_query(question)
    query_dict = {
        "time_period": time_period,
        "location": location,
        "list_keyword": list_keyword,
        "weekday": weekday,
        "time_filter": time_filter
    }

    filters = construct_filter(query_dict)
    query = {
        "_source": ['event_id', 'ImageID', 'local_time', 'long_description', 'city', 'time_diff', 'semantic_name',
                    'medium_hour_diff', 'day_of_week'],
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "long_description": processed_query
                    }
                },
                "filter": filters
            }
        }
    }
    json_query = json.dumps(query)
    response = requests.post(url, data=json_query, headers={'Content-Type': 'application/json'})
    return response.json()


def ask_llm(prompt):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer the question based on the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content


def create_prompt(question, relevant_document):
    hits = relevant_document['hits']['hits']
    prompt = f'Answer this question {question} based on the provided information with short explaination: \n'
    for hit in hits:
        prompt += f"Image id {hit['_source']['ImageID']}: {hit['_source']['long_description']} at" \
                  f" {hit['_source']['local_time']} in {hit['_source']['city']} \n "

    return prompt


def question_classification(question: str):
    # Classify the question to visual or metadata related question
    # Input: question
    # Output: type = 0 -> visual question, type = 1 -> metadata question, type = -1: unknown
    question_lower = question.lower()

    visual_keywords = ["what", "who", "why"]
    metadata_keywords = ["where", "when", "how many", "how much", "what date", "what did", 'which']

    for keyword in metadata_keywords:
        if keyword in question_lower:
            return 1
    for keyword in visual_keywords:
        if keyword in question_lower:
            return 0
    return -1


def extract_question_component(question_query):
    # Process the question to the question, the context in the question and the question to confirm the content of image
    # Input: Question, free text
    # Output: context_query, question_to_ask, question_to_confirm

    # Define values for lexical properties
    question = ['WDT', 'WP', 'WP$', 'WRB']
    verb = ['VBZ', 'VBP', 'VBN', 'VBG', "VBD", 'VB']
    noun = ['NNS', 'NNPS', 'NNP', 'NN']
    dot_comma = [',', '.']
    context = []
    question_word = []
    question_verb = []
    question_context = []
    unknown = []
    flag = 'context'

    # Tokenize the sentences to word by word
    tokenizer = WordPunctTokenizer()
    question_query = tokenizer.tokenize(question_query)
    # Add tags to word
    tags = pos_tag(question_query)
    question_index = 0

    for index in range(len(tags)):
        # Add words to the question follow these rules, else context
        if tags[index][1] in question:
            question_word.append(tags[index])
            flag = 'question'
            question_index = index
        elif flag == 'question' and tags[index][1] in noun and (question_index + 2) >= index:
            question_word.append(tags[index])
        elif flag == 'question' and tags[index][1] in verb and len(question_verb) == 0:
            question_verb.append(tags[index])
        elif tags[index][1] in dot_comma:
            try:
                if tags[index + 1][1] in question:
                    flag = 'question'
            except:
                flag = 'context'
        elif flag == 'question':
            question_context.append(tags[index])
        elif flag == 'context':
            context.append(tags[index])
        else:
            unknown.append(tags[index])

    # Create different question and context to ask and confirm
    context_query = question_context + context
    context_query_return = ''
    for cxt in context_query:
        context_query_return += (' ' + cxt[0])

    question_to_ask = question_word + question_verb + question_context + context
    question_to_ask_return = ''
    for cxt in question_to_ask:
        if cxt[1] != 'CD':
            question_to_ask_return += (' ' + cxt[0])

    question_to_confirm = question_context + context
    if len(question_verb) == 0:  # Question start with a verb
        question_to_confirm_return = ''
    else:
        if question_verb[0][0] in ['is', 'be', 'are', 'am', 'was', 'were']:
            question_to_confirm_return = 'Was'
        else:
            question_to_confirm_return = 'Did'
    for cxt in question_to_confirm:
        if cxt[1] != 'CD':
            question_to_confirm_return += (' ' + cxt[0])
    question_to_confirm_return += ' in the images?'
    return context_query_return, question_to_ask_return, question_to_confirm_return


def RAG(question):
    # retrieve all data
    relevant_document = rag_retriever(question, 50)
    retrieved_result = []

    for hit in relevant_document['hits']['hits']:
        source = hit['_source']
        extracted_source = {
            'ImageID': source['ImageID'],
            'new_name': source['semantic_name'],
            'city': source['city'],
            'event_id': source['event_id'],
            'local_time': source['local_time'],
            'day_of_week': source['day_of_week']
        }
        retrieved_result.append({
            "_index": hit["_index"],
            "_id": hit["_id"],
            "_score": hit["_score"],
            "_source": extracted_source
        })
    retrieved_result = [{'current_event': each_result} for each_result in retrieved_result]
    retrieved_result = add_image_link(retrieved_result)
    # Create prompt
    prompt = create_prompt(question, relevant_document)
    logging.info(f'RAG: prompt for LLM: {prompt}')
    # print(prompt, '\n ')
    # Ask LLM
    answer = ask_llm(prompt)
    logging.info(f'RAG: answer{answer}')
    return answer, retrieved_result


def rag_question_answering(query):
    # This function is to answer a question based on the caption generated by BLIP2. It use RAG to retrieve the relevant
    # information and a LLM reader to generate the answer for the question.
    # Input: A question end with ? mark
    # Output: a textual answer

    # Extract different component of the query
    logging.info(f'QA: Query processing for query: {query}')
    context, question, question_confirm = extract_question_component(query)
    # Classify the type of question, visual or non-visual. This is for old method, now is no branching
    question_type = question_classification(query)
    logging.info(f"QA: Classified question type: {question_type}")

    # Perform rag on the question
    textual_answer, retrieved_result = RAG(question)
    logging.info("QA: Finish QA")
    # answer_dict = {}
    # retrieved_context = ''
    # count = 0
    # for result in retrieved_results['hits']['hits']:
    #     # Get the image content to analyse
    #     image_id = result['_source']['ImageID']
    #     image_name, year_month, day = extract_date_imagename(image_id)
    #     image_path = settings.image_directory + '/' + year_month + '/' + day + '/' + image_name + ".webp"
    #     raw_image = Image.open(image_path).convert('RGB')
    #     image = instruct_vis_processor["eval"](raw_image).unsqueeze(0).to(device)
    #     # Confirm if the image is relevant
    #     logging.info("QA: Confirm the retrieve result is correct or not")
    #     confirm = instruct_model.generate({"image": image,
    #                                        "prompt": f"Based on the provided images, "
    #                                                  f"answer this question {question_confirm}. Answer: "})
    #     if ~('no' in confirm[0].lower() or 'wrong' in confirm[0].lower()):
    #         if question_type == 0:
    #             # BLIP-2 for visual related question
    #             answer = instruct_model.generate({"image": image,
    #                                               "prompt": f"Based on the provided images, "
    #                                                         f"answer this question {query}. "
    #                                                         f"Answer: "})
    #             answer_dict[image_id] = answer[0]
    #             logging.info(f"QA: visual question result: {answer_dict}")
    #         elif question_type == 1:
    #             # Create data dict for LLM
    #             # answer_dict['event_' + str(count)] = {}
    #             # answer_dict['event_' + str(count)]['time'] = result['_source']['local_time']
    #             # answer_dict['event_' + str(count)]['city'] = result['_source']['city']
    #             # answer_dict['event_' + str(count)]['location'] = result['_source']["new_name"]
    #             # answer_dict['event_' + str(count)]['event'] = context
    #
    #             datetime = result['_source']['local_time']
    #             city = result['_source']['city']
    #             semantic_name = result['_source']["new_name"]
    #             retrieved_context += f'{datetime}, {context}, at {semantic_name}, {city} \n'
    #
    #             count += 1
    #         else:
    #             return ['Unknown question type']
    # if len(retrieved_context) > 0 and question_type == 1:
    #     response = openai.ChatCompletion.create(
    #         model='gpt-3.5-turbo',
    #         messages=[
    #             {'role': 'user',
    #              'content': f"Base on the provided data {retrieved_context}. Answer this question {query}: "}
    #         ]
    #     )
    #     answer = response['choices'][0]['message']['content']
    #     answer_dict['answer'] = answer
    #     logging.info(f"QA: context question result: {answer_dict}")
    return retrieved_result, textual_answer

# def answer_aggregation(result_dict):
#     # Aggregate the results from LLM models, to get relevant results
#     if 'answer' in result_dict:
#         # the result for metadata related question
#         return result_dict['answer']
#     else:
#         # the result for visual related question
#         list_answer = list(result_dict.values())
#         final_result = {item: list_answer.count(item) for item in list_answer}
#         return final_result
