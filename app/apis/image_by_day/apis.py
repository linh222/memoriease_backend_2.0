import ast

import pandas as pd
from fastapi import APIRouter, status

from app.apis.image_by_day.schema import FeatureModelImage
from app.config import root_path, IMAGE_SERVER, AWS_SECRET_KEY, AWS_ACCESS_KEY, BUCKET
import boto3
from io import StringIO
router = APIRouter()

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
response = s3.get_object(Bucket=BUCKET, Key='image_by_event_for_visualization1.csv')
csv_data = response['Body'].read().decode('utf-8')
df_image = pd.read_csv(StringIO(csv_data))


def generate_image_link(image_id):
    year_month = image_id[:6]
    day = image_id[6:8]
    return f"{IMAGE_SERVER}/{year_month}/{day}/{image_id}.webp"


def add_link(row):
    row['image_link'] = generate_image_link(row['ImageID'])
    similar_images = ast.literal_eval(row['image_event'])
    row['image_event'] = [generate_image_link(image) for image in similar_images]
    row['ImageID'] += '.jpg'
    row['local_time'] = row['datetime'].replace(' ', 'T')
    return row


@router.post("/image", status_code=status.HTTP_200_OK)
async def get_image(feature: FeatureModelImage):
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
