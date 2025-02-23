import asyncio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, concurrency
from data_pipeline.schemas import InferSchema
from data_pipeline.services import convert_file_to_df, preprocess_dataset

from SystemManagement.enums import SystemRole
from app.common.authentication import protected_route
from app.common.db_collections import Collections
from app.common.dependencies import AuthCredentialDepend
import json
import logging
from app.common.websocket import WebsocketEventResult
from app.common.db_connector import client

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/preprocess-dataset", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def preprocess(
    CREDENTIALS: AuthCredentialDepend,
    mapping: str = Form(),
    file: UploadFile = File(...),
    CURRENT_USER=None,
):
    try:
        mapping = json.loads(mapping)
        if isinstance(mapping, str):
            mapping = json.loads(mapping)
            
        db_str, cur_user_id = CURRENT_USER.get("db"), CURRENT_USER.get("_id")

        features = mapping.get("features")
        label = mapping.get("label")
        
        df = await convert_file_to_df(file)
        return await preprocess_dataset(df, features, label, db_str, cur_user_id)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        if isinstance(e, Exception):
            raise HTTPException(400, str(e))
        


