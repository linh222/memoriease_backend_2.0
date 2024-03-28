import ast
from io import StringIO

import boto3
import pandas as pd
from fastapi import APIRouter, status

from app.apis.image_by_day.schema import FeatureModelImage
from app.config import IMAGE_SERVER, AWS_SECRET_KEY, AWS_ACCESS_KEY, BUCKET, IMAGE_EXT

router = APIRouter()

# TODO: add visualize for lsc20
# Load data from s3. the event segmentation for visual image by day -> day -> event -> image
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
response = s3.get_object(Bucket=BUCKET, Key='image_for_visualization_no_event.csv')
csv_data = response['Body'].read().decode('utf-8')
df_image = pd.read_csv(StringIO(csv_data))


def generate_image_link(image_id, IMAGE_SERVER):
    # Create the image link from image id
    # Input: image_id: 20190101_202020
    # Output: image link: http://localhost:9200/201901/01/20190101_202020.webp
    year_month = image_id[:6]
    day = image_id[6:8]
    return f"{IMAGE_SERVER}/{year_month}/{day}/{image_id}.{IMAGE_EXT}"


def add_link(row):
    # Post process to add image link, create list of similar images, ImageID and local time
    row['image_link'] = generate_image_link(row['ImageID'], IMAGE_SERVER)
    similar_images = ast.literal_eval(row['image_event'])
    row['image_event'] = [generate_image_link(image, IMAGE_SERVER) for image in similar_images]
    row['ImageID'] += '.jpg'
    row['local_time'] = row['datetime'].replace(' ', 'T')
    return row


@router.post("/image", status_code=status.HTTP_200_OK)
async def get_image(feature: FeatureModelImage):
    # API endpoint to get image by date
    # Input: day_month_year: 2019-01-01, time_period: morning, hour: 1-> 24
    # Output: all image in the filtered space.

    date = feature.day_month_year
    time_period = feature.time_period
    hour = feature.hour

    conditions = (df_image['date'] == date)  # Predefine filter condition
    if time_period:
        conditions &= (df_image['time_period'] == time_period)  # add time filter if time filter != ''
    if hour:
        conditions &= (df_image['hour'] == int(hour))  # add hour filter if hour filter != ''

    # List of selected data
    selected_columns = ['ImageID', 'image_event', 'datetime', 'day_of_week']
    # Generate temp df with filter conditions
    temp = df_image.loc[conditions, selected_columns].reset_index(drop=True)
    # Add link to the main image, the similar image
    output_list = temp.apply(add_link, axis=1).to_dict(orient='records')

    return output_list


def include_router(app):
    app.include_router(router)
