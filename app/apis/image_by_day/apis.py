import ast

import pandas as pd
from fastapi import APIRouter, status

from app.apis.image_by_day.schema import FeatureModelImage
from app.config import root_path, IMAGE_SERVER

router = APIRouter()
df_image = pd.read_csv('{}/app/models/image_by_event_for_visualization1.csv'.format(root_path))


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

# def add_link(df):
#     output_list = []
#     for i, row in df.iterrows():
#         imageid = row['ImageID']
#         year_month = imageid[:6]
#         day = imageid[6:8]
#         image_name = imageid
#         row['image_link'] = IMAGE_SERVER + '/{}/{}/{}.webp'.format(year_month, day, image_name)
#         similar_image = ast.literal_eval(row['image_event'])
#         row['image_event'] = []
#         if len(similar_image) > 0:
#             for image in similar_image:
#                 year_month = image[:6]
#                 day = image[6:8]
#                 image_name = image
#                 row['image_event'].append(IMAGE_SERVER + '/{}/{}/{}.webp'.format(year_month, day, image_name))
#         temp_dict = {'ImageID': row['ImageID'] + '.jpg', 'similar_images': row['image_event'],
#                      'local_time': row['datetime'].replace(' ', 'T'), 'day_of_week': row['day_of_week'],
#                      'image_link': row['image_link']}
#         output_list.append(temp_dict)
#     return output_list
#
#
# @router.post(
#     "/image",
#     status_code=status.HTTP_200_OK
# )
# async def get_image(feature: FeatureModelImage):
#     date = feature.day_month_year
#     time_period = feature.time_period
#     hour = feature.hour
#     if time_period == '' and hour == '':
#         temp = df_image[df_image['date'] == date][['ImageID', 'image_event',
#                                                    'datetime', 'day_of_week']].reset_index(drop=True)
#     elif time_period != '' and hour == '':
#         temp = df_image.loc[((df_image['date'] == date) & (df_image['time_period'] == time_period))][
#             ['ImageID', 'image_event', 'datetime', 'day_of_week']].reset_index(drop=True)
#     elif time_period == '' and hour != '':
#         temp = df_image.loc[((df_image['date'] == date) & (df_image['hour'] == int(hour)))][
#             ['ImageID', 'image_event', 'datetime', 'day_of_week']].reset_index(drop=True)
#     else:
#         temp = df_image.loc[
#             ((df_image['date'] == date) & (df_image['hour'] == int(hour) &
#             (df_image['time_period'] == time_period)))][
#             ['ImageID', 'image_event', 'datetime', 'day_of_week']].reset_index(drop=True)
#     output_list = add_link(temp)
#
#     return output_list
