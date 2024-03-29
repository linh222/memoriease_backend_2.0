import openai
import logging
from PIL import Image
from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer

from app.apis.api_utils import extract_date_imagename
from app.config import settings
from app.predictions.predict import retrieve_image

logging.basicConfig(filename='memoriease_backend.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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


def process_question(question_query):
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


def process_result(query, blip2_embed_model, blip2_txt_processor,
                   instruct_model, instruct_vis_processor, device):
    # Extract different component of the query
    context, question, question_confirm = process_question(query)
    question_type = question_classification(query)
    logging.info(f"Classified question type: {question_type}")
    # different return length for different question type, the retriever retrieve results for event
    if question_type == 0:
        retrieved_results = retrieve_image(concept_query=context, embed_model=blip2_embed_model,
                                           txt_processor=blip2_txt_processor, size=5)
    else:
        retrieved_results = retrieve_image(concept_query=context, embed_model=blip2_embed_model,
                                           txt_processor=blip2_txt_processor, size=30)
    logging.info("QA: Event retriever finished")
    answer_dict = {}
    retrieved_context = ''
    count = 0
    for result in retrieved_results['hits']['hits']:
        # Get the image content to analyse
        image_id = result['_source']['ImageID']
        image_name, year_month, day = extract_date_imagename(image_id)
        image_path = settings.image_directory + '/' + year_month + '/' + day + '/' + image_name + ".webp"
        raw_image = Image.open(image_path).convert('RGB')
        image = instruct_vis_processor["eval"](raw_image).unsqueeze(0).to(device)
        # Confirm if the image is relevant
        logging.info("QA: Confirm the retrieve result is correct or not")
        confirm = instruct_model.generate({"image": image,
                                           "prompt": f"Based on the provided images, "
                                                     f"answer this question {question_confirm}. Answer: "})
        if ~('no' in confirm[0].lower() or 'wrong' in confirm[0].lower()):
            if question_type == 0:
                # BLIP-2 for visual related question
                answer = instruct_model.generate({"image": image,
                                                  "prompt": f"Based on the provided images, "
                                                            f"answer this question {query}. "
                                                            f"Answer: "})
                answer_dict[image_id] = answer[0]
                logging.info(f"QA: visual question result: {answer_dict}")
            elif question_type == 1:
                # Create data dict for LLM
                # answer_dict['event_' + str(count)] = {}
                # answer_dict['event_' + str(count)]['time'] = result['_source']['local_time']
                # answer_dict['event_' + str(count)]['city'] = result['_source']['city']
                # answer_dict['event_' + str(count)]['location'] = result['_source']["new_name"]
                # answer_dict['event_' + str(count)]['event'] = context

                datetime = result['_source']['local_time']
                city = result['_source']['city']
                semantic_name = result['_source']["new_name"]
                retrieved_context += f'{datetime}, {context}, at {semantic_name}, {city} \n'

                count += 1
            else:
                return ['Unknown question type']
    if len(retrieved_context) > 0 and question_type == 1:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user',
                 'content': f"Base on the provided data {retrieved_context}. Answer this question {query}: "}
            ]
        )
        answer = response['choices'][0]['message']['content']
        answer_dict['answer'] = answer
        logging.info(f"QA: context question result: {answer_dict}")
    return answer_aggregation(answer_dict)


def answer_aggregation(result_dict):
    # Aggregate the results from LLM models, to get relevant results
    if 'answer' in result_dict:
        # the result for metadata related question
        return result_dict['answer']
    else:
        # the result for visual related question
        list_answer = list(result_dict.values())
        final_result = {item: list_answer.count(item) for item in list_answer}
        return final_result
