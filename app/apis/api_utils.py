import logging
import time

import pandas as pd
from fastapi import Request

from app.config import IMAGE_SERVER, root_path
from app.predictions.utils import process_query

df_event = pd.read_csv('{}/app/models/event_segmentation.csv'.format(root_path),
                       dtype={"path": 'str', 'event_id': 'int64'})


def add_image_link(results):
    if len(results) >= 1:
        for result in results:
            image_id = result['current_event']['_source']['ImageID']
            year_month = image_id[:6]
            day = image_id[6:8]
            image_name = image_id[0:-4]
            result['current_event']['_source'][
                'image_link'] = IMAGE_SERVER + '/{}/{}/{}.webp'.format(year_month, day,
                                                                       image_name)
            # similar images
            list_similar_image = []
            event_id = int(result['current_event']['_source']['event_id'])
            sim_image = df_event[df_event['event_id'] == event_id]['path'].values
            if sim_image.shape[0] > 1:
                for img in range(sim_image.shape[0]):
                    image_id = sim_image[img]
                    if image_id != image_name:
                        year_month = image_id[:6]
                        day = image_id[6:8]
                        list_similar_image.append(
                            '{}/{}/{}/{}.webp'.format(IMAGE_SERVER, year_month, day, image_id))
            result['current_event']['_source']['similar_images'] = list_similar_image
    return results


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
