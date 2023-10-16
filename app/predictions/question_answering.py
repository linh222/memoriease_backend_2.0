import openai
from PIL import Image
from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer

from app.apis.api_utils import extract_date_imagename
from app.config import settings
from app.predictions.predict import retrieve_image


def question_classification(question: str):
    # type = 0 -> visual question, type = 1 -> metadata question, type = -1: unknown
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

    question = ['WDT', 'WP', 'WP$', 'WRB']
    verb = ['VBZ', 'VBP', 'VBN', 'VBG', "VBD", 'VB']
    noun = ['NNS', 'NNPS', 'NNP', 'NN']
    adj = ['JJS', 'JJR', 'JJ']
    md = ['MD']
    dot_comma = [',', '.']
    context = []
    question_word = []
    question_verb = []
    question_context = []
    unknown = []
    flag = 'context'

    tokenizer = WordPunctTokenizer()
    question_query = tokenizer.tokenize(question_query)
    tags = pos_tag(question_query)
    question_index = 0
    for index in range(len(tags)):
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

    context_query = question_context + context
    context_query_return = ''
    for cxt in context_query:
        context_query_return += (' ' + cxt[0])

    question_to_ask = question_word + question_verb + question_context + context
    question_to_ask_return = ''
    for cxt in question_to_ask:
        if cxt[1] != 'CD':
            question_to_ask_return += (' ' + cxt[0])

    return context_query_return, question_to_ask_return


def process_result(query, semantic_name, start_hour, end_hour, is_weekend, blip2_embed_model, blip2_txt_processor,
                   instruct_model, instruct_vis_processor, device):
    context, question = process_question(query)
    retrieved_results = retrieve_image(concept_query=context, embed_model=blip2_embed_model,
                                       txt_processor=blip2_txt_processor, semantic_name=semantic_name,
                                       start_hour=start_hour, end_hour=end_hour, is_weekend=is_weekend, size=10)
    question_type = question_classification(query)

    if question_type == 0:
        # Process the visual related question. Assume that the blip2/instructblip is already load
        answer_dict = {}
        for result in retrieved_results['hits']['hits']:
            image_id = result['_source']['ImageID']
            image_name, year_month, day = extract_date_imagename(image_id)
            image_path = settings.image_directory + '/' + year_month + '/' + day + '/' + image_name + ".webp"
            raw_image = Image.open(image_path).convert('RGB')
            image = instruct_vis_processor["eval"](raw_image).unsqueeze(0).to(device)
            answer = instruct_model.generate({"image": image,
                                              "prompt": f"Based on the provided images, "
                                                        f"answer this question {query}. Answer: "})
            answer_dict[image_id] = answer
        return answer_aggregation(answer_dict)
    elif question_type == 1:
        # Process the metadata related question. Retrieve the metadata (time, location, city,)
        metadata_dict = {}
        for index, result in enumerate(retrieved_results['hits']['hits']):
            metadata_dict['event_' + str(index)] = {}
            metadata_dict['event_' + str(index)]['time'] = result['_source']['local_time']
            metadata_dict['event_' + str(index)]['city'] = result['_source']['city']
            metadata_dict['event_' + str(index)]['location'] = result['_source']["new_name"]
            metadata_dict['event_' + str(index)]['event'] = context
        # Call chatgpt to answer
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user',
                 'content': f"Base on the provided data {metadata_dict} in dictionary for with each key is each "
                            f"event. Answer this question {query} by return a dictionary of key is answer and value is "
                            f"the number of time that key in data."}
            ]
        )
        answer = response['choices'][0]['message']['content']
        metadata_dict['answer'] = answer
        return answer_aggregation(metadata_dict)

    else:
        return ['Unknown question type']


def answer_aggregation(result_dict):
    if 'answer' in result_dict:
        #the result for metadata related question
        return result_dict['answer']
    else:
        # the result for visual related question
        list_answer = list(result_dict.values())
        return {item: list_answer.count(item) for item in list_answer}


if __name__ == "__main__":
    question1 = "What is the colour of the jacket worn by the black and white panda-bear toy that can sometimes be " \
                "seen with the two long rabbits"
    question2 = "What was the name of the car service/repair company that I used in the summer of 2019? I want to " \
                "call them to get my car fixed"
    question3 = "I visited a famous vietnam temple with friends a few years ago. What did I do immediately afterwards?"
    question4 = "I can't find my hand drill / electric screwdriver.  when was I last using it before 1st Jupy 2020?"
    question5 = "I normally wear shirts, but what is the brand of the grey t-shirt that I wore at the start of " \
                "covid-time?"
    question6 = "On what date in 2019 did I go homewares shopping around midnight in Ireland?"
    question7 = "Which airline did I fly with most often in 2019?"
    question8 = "I don’t often go to the cinema, but I went to see the ‘Joker’ in 2019. What date was that?"
    question9 = "What airline did I fly on for my first flight in 2020? I remember it was a small plane, perhaps an " \
                "ATR-72."
    question10 = "I had some Strawberry Jam / Preserve in my refrigerator. It was the best jam I ever tasted. What " \
                 "brand was it?"
    question11 = "I used to have a red and white Christmas mug. What was written on the mug"
    question12 = "What date did I eat pancakes with cherries and strawberries for breakfast outside?"
    question13 = "On what date did I meet Klaus Schoeffmann first in 2020?"
    question14 = "How many time did I have dinner at a restaurant in 2019"

    # import torch
    # from LAVIS.lavis.models import load_model_and_preprocess
    #
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
    #                                                                   model_type="coco", is_eval=True,
    #                                                                   device=device)
    # print("cuda" if torch.cuda.is_available() else "cpu")
    # answer = process_result(question6, embed_model=model, txt_processor=txt_processors)
    # print(answer)

    # print(f"Question 1: {question_classification(question1)}")
    # print(f"Question 2: {question_classification(question2)}")
    # print(f"Question 3: {question_classification(question3)}")
    # print(f"Question 4: {question_classification(question4)}")
    # print(f"Question 5: {question_classification(question5)}")
    # print(f"Question 6: {question_classification(question6)}")
    # print(f"Question 7: {question_classification(question7)}")
    # print(f"Question 8: {question_classification(question8)}")
    # print(f"Question 9: {question_classification(question9)}")
    # print(f"Question 10: {question_classification(question10)}")
    # print(f"Question 11: {question_classification(question11)}")
    # print(f"Question 12: {question_classification(question12)}")
    # print(f"Question 13: {question_classification(question13)}")
    # print(f"Question 14: {question_classification(question14)}")
