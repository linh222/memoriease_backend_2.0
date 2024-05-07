import logging
import time
from io import StringIO

import boto3
import pandas as pd
from fastapi import Request

from app.config import IMAGE_SERVER, root_path, AWS_SECRET_KEY, AWS_ACCESS_KEY, BUCKET, IMAGE_EXT
from app.predictions.utils import process_query

# Load resource from s3 for event segmentation
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
response = s3.get_object(Bucket=BUCKET, Key='event_segmentation_lsc20_lsc23.csv')
csv_data = response['Body'].read().decode('utf-8')
df_event = pd.read_csv(StringIO(csv_data), dtype={"path": 'str', 'event_id': 'int64'})


def add_image_link(results):
    # add image link for the final output
    if len(results) >= 1:
        for result in results:
            image_id = result['current_event']['_source']['ImageID']
            image_name, year_month, day = extract_date_imagename(image_id)
            result['current_event']['_source'][
                'image_link'] = IMAGE_SERVER + '/{}/{}/{}.{}'.format(year_month, day,
                                                                     image_name, IMAGE_EXT)
            # similar images
            list_similar_image = []
            try:
                event_id = int(result['current_event']['_source']['event_id'])
                sim_image = df_event[df_event['event_id'] == event_id]['path'].values
                if sim_image.shape[0] > 1:
                    for img in range(sim_image.shape[0]):
                        image_id = sim_image[img]
                        if image_id != image_name:
                            image_name, year_month, day = extract_date_imagename(image_id)
                            list_similar_image.append(
                                '{}/{}/{}/{}.{}'.format(IMAGE_SERVER, year_month, day, image_name, IMAGE_EXT))
                result['current_event']['_source']['similar_images'] = list_similar_image
            except Exception as e:
                result['current_event']['_source']['similar_images'] = []
    return results


def extract_date_imagename(image_id):
    if image_id.find('/') == -1:
        year_month = image_id[:6]
        day = image_id[6:8]
        image_name = image_id
    else:
        date, image_name = image_id.split('/')
        year_month = "".join(date.split("-")[:2])
        day = date.split("-")[-1]
    return image_name.replace('.jpg', ''), year_month, day


def logger_init():
    logger = logging.getLogger("request_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("{}/app/evaluation_model/request_log.txt".format(root_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, formatter, file_handler


logger, formatter, file_handler = logger_init()


class RequestTimestampMiddleware:
    def __init__(self, app, router_path):
        self.app = app
        self.router_path = router_path

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            if request.url.path.startswith(self.router_path):
                logger.info(f"Request received to {request.url.path} at {request.client.host} in {int(time.time())}")
        await self.app(scope, receive, send)


def metadata_logging2file(query, topic):
    timestamp = int(time.time())
    processed_text, list_keyword, time_period, weekday, time_filter, location = process_query(query)
    metadata = f"{processed_text}:time_period={time_period}:weekday={weekday}:time={str(time_filter)}" \
               f":location={location}"

    with open("{}/app/evaluation_model/metadata_log.txt".format(root_path), "a") as file:
        file.write("\n" + "{},{},{},{}".format(timestamp, topic, 'query_string', metadata))
