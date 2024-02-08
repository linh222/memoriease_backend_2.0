# TODO Develop a function to chat
from app.predictions.autofilter_construction import construct_filter, retrieve_result
from openai import OpenAI
import openai
from dotenv import load_dotenv
from app.config import root_path
import os
import logging


logging.basicConfig(filename='conversational_search_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(str(root_path) + '/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


def textual_answer(query):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Acting as you are a multi-modal lifelog retrieval assistant that can display or search "
                           "content and you have already found the results from the user query and displayed lifelog "
                           "images and information in the interface. You have full access to the system and you "
                           "already displayed all of the images from the users query on the right side of the "
                           "interface already. Give an answer that you already found the results from the user query "
                           "and showed on the interface.\nHere are some examples and you should follow "
                           "strictly:\nUsers: Find the moment when I reached the edge of a lake. It was a cold day "
                           "spring in Wicklow in 2019. Before that, I have a short relaxing walk.\nAssistant: I have "
                           "found all the moments when you reached the edge of a lake on a cold spring day in Wicklow "
                           "in 2019. Before that, you had a short relaxing walk. Please take a look at the images on "
                           "the right side of the display to revisit those memories. If you need more details or "
                           "specific information about any particular moment, feel free to ask!\nUsers: I was in a "
                           "cinema\nAssistant: I apologize for the confusion. It seems that I may have misunderstood "
                           "your query. Unfortunately, I do not have access to the specific details of your past "
                           "activities such as going to the cinema. I can only display and search the content that is "
                           "available in the lifelog database. If you have any other queries or if there's anything "
                           "else I can assist you with, please let me know."
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


def formulate_previous_chat(previous_chat: list):
    formatted_chat = ''
    for i in range(len(previous_chat)):
        formatted_chat += f'Turn {i + 1}: {previous_chat[i]}\n'
    return formatted_chat


def chatgpt_verify_query(previous_query, current_query):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Acting as a chat assistant for a lifelog retrieval system. A user has asked for "
                           "information about lifelog with the previous query and current query."
            },
            {
                "role": "user",
                "content": "If the current query still follows the previous query, consolidate the two queries into a "
                           "single query and return a json file with format {'same_topic'=True, query: ''}, "
                           "else return json format {'same_topic'=False}\nExample:\nTurn 1: Find all the images I was "
                           "in a pub in Ireland.\nCurrent query: I remember that is in 2020\nResult: {"
                           "'same_topic'=True, query: 'Find all the images I was in a pub in Ireland in 2020'}\n\nDo "
                           "the same for this query:\nPrevious query: Find all the images I sitting on a chair in a "
                           "park\nCurrent query: in Dublin in 2019"
            },
            {
                "role": "assistant",
                "content": "{'same_topic': True, 'query': 'Find all the images I sit on a chair in a park in Dublin "
                           "in 2019'}"
            },
            {
                "role": "user",
                "content": "That is true. You are great.\nHow about this: \nTurn 1: Find all the images I sitting on "
                           "a chair in a park\nCurrent query: I was on a library in Dublin City University on Monday"
            },
            {
                "role": "assistant",
                "content": "{'same_topic': False}"
            },
            {
                "role": "user",
                "content": "How about this:\nTurn 1: Find all the images I sitting on a chair in a park\nCurrent "
                           "query: on Monday"
            },
            {
                "role": "assistant",
                "content": "{'same_topic': True, 'query': 'Find all the images I sit on a chair in a park on Monday'}"
            },
            {
                "role": "user",
                "content": "Turn 1: Find all the images I was in a pub in Ireland\nTurn 2: I remember that was in "
                           "2020\nCurrent query: I remember I drank Guinness in the pub"
            },
            {
                "role": "assistant",
                "content": "{'same_topic': True, 'query': 'Find all the images I was in a pub in Ireland in 2020. I "
                           "drank Guinness in the pub.'}"
            },
            {
                "role": "user",
                "content": f"{previous_query}Current query: {current_query}"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


def chat(query: str, previous_chat: list, model, txt_processors):
    # Function to perform a chat with the model
    # Input:
    #   Query: The current query from users
    #   Previous chat: the previous queries from users
    logging.info(f'Received query {query} with previous chat {previous_chat}')
    if query == '':
        raise ValueError('Empty string')
    if len(previous_chat) == 0:
        logging.info('First round search')
        # Perform first time search
        filters, main_event, previous_event, after_event = construct_filter(query)
        logging.info(f'Extracted information: filters: {filters}, main_event: {main_event},'
                     f' previous_event: {previous_event}, after_event: {after_event}')
        result = retrieve_result(main_event_context=main_event, previous_event_context=previous_event,
                                 after_event_context=after_event,
                                 filters=filters, embed_model=model, txt_processor=txt_processors, size=100)
        return_answer = 'I am so sorry but I cannot find any relevant information about your query. Please refine ' \
                        'your query to make it more specifically.'
        if result is not None:
            if len(result) > 0:
                return_answer = textual_answer(query)
        logging.info(f'Answer: {return_answer}')
    else:

        # perform search after several rounds
        # Step 1: if the current query and previous query are in the same topic -> create a united query
        #         Else: Act as first time request
        logging.info('Multi round search')
        formatted_previous_chat = formulate_previous_chat(previous_chat)
        response_verify_query = chatgpt_verify_query(previous_query=formatted_previous_chat, current_query=query)
        response_verify_query = eval(response_verify_query.choices[0].message.content)
        if response_verify_query['same_topic']:
            retrieving_query = response_verify_query['query']
        else:
            retrieving_query = query
        # print(response_verify_query)
        # Step 2: Perform query extractor
        filters, main_event, previous_event, after_event = construct_filter(retrieving_query)
        logging.info(f'Extracted information: filters: {filters}, main_event: {main_event},'
                     f' previous_event: {previous_event}, after_event: {after_event}')
        result = retrieve_result(main_event_context=main_event, previous_event_context=previous_event,
                                 after_event_context=after_event,
                                 filters=filters, embed_model=model, txt_processor=txt_processors, size=100)
        # Step 3: Ask for response
        return_answer = 'I am so sorry but I cannot find any relevant information about your query. Please refine ' \
                        'your query to make it more specifically.'
        if result is not None:
            if len(result) > 0:
                return_answer = textual_answer(retrieving_query)
        logging.info(f'Answer: {return_answer}')
    return result, return_answer
