from fastapi import APIRouter, status

from app.config import root_path
from .schemas import ResponseModel

router = APIRouter()


@router.post(
    "/metadata",
    status_code=status.HTTP_200_OK,
)
async def get_metadata_submission(feature: ResponseModel):
    try:
        with open(f'{root_path}/app/evaluation_model/request_log.txt') as file:
            data = file.readlines()
            submit_time = data[-1][-11:]

        timestamp = feature.timestamp
        metadata = feature.metadata
        response = feature.response
        topic = feature.topic

        if response == 'Correct':
            with open(f'{root_path}/app/evaluation_model/ntcir_interactive_logging.csv', 'a') as file:
                file.write(f'DCU,MEMORIEASE_SAT01,,{metadata},{int(timestamp) // 1000 - int(submit_time)},1\n')
        return True
    except:
        ValueError('Fail to read and write metadata')


def include_router(app):
    app.include_router(router)
