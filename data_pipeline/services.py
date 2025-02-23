import asyncio
import json
import re
import time
from typing import List, Tuple
from bson import ObjectId
from fastapi import UploadFile
import pandas as pd
import py_vncorenlp
from pymongo import MongoClient, errors
import concurrent.futures
import math
import os
import logging
import torch
import threading
from functools import lru_cache
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from app.common.db_collections import Collections
from app.common.utils import get_current_datetime
from app.common.db_connector import client
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`",
)

MODEL = "vinai/phobert-base"
URL_PATTERN = re.compile(r'https?://\S+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+", 
    flags=re.UNICODE
)
WHITESPACE_PATTERN = re.compile(r'\s+')

tokenizer = AutoTokenizer.from_pretrained(MODEL)

@lru_cache(maxsize=None)
def get_rdrsegmenter():
    return py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], 
        save_dir=os.environ.get("VNCORENLP_PATH")
    )

rdrsegmenter = get_rdrsegmenter()

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

def preprocess_data(df: pd.DataFrame, cols: List[str], max_length: int = 256) -> Tuple[dict, list]:
    # Batch preprocessing
    texts = df[cols[:-1]].apply(
        lambda row: ' </s> '.join(row.values.astype(str)) + ' </s>',
        axis=1
    ).tolist()

    encodings = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        return_token_type_ids=False
    )
    
    return encodings, df[cols[-1]].values

def get_optimized_dataloaders(
    df: pd.DataFrame,
    cols: List[str],
    batch_size: int = 128,
    max_length: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    # Stratified split
    label_col = cols[-1]
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=42
    )

    train_encodings, train_labels = preprocess_data(train_df, cols, max_length)
    val_encodings, val_labels = preprocess_data(val_df, cols, max_length)
    test_encodings, test_labels = preprocess_data(test_df, cols, max_length)

    train_dataset = PreprocessedDataset(train_encodings, train_labels)
    val_dataset = PreprocessedDataset(val_encodings, val_labels)
    test_dataset = PreprocessedDataset(test_encodings, test_labels)

    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = (1.0 / (class_counts + 1e-6)) * len(train_labels) / len(class_counts)
    class_weights = class_weights.float()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        # drop_last=True
    )

    return train_loader, val_loader, test_loader, class_weights


def subset_to_dataframe(subset, original_dataframe):
    indices = subset.indices
    return original_dataframe.iloc[indices].reset_index(drop=True)


def text_cleaning(text: str) -> str:
    if not text:
        return ""
    text = URL_PATTERN.sub('', text)
    text = EMOJI_PATTERN.sub('', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


async def preprocess_dataset(
    df: pd.DataFrame, features: List[str], label: str, db_str: str, cur_user_id: str
):
    df[label] = pd.to_numeric(df[label], errors='coerce')
    filtered_df = df.dropna(subset=[label, *features]).copy()
    
    filtered_df[features] = filtered_df[features].astype(str)
    
    df_features = filtered_df[features]
    df_labels = filtered_df[label]

    logger.info(f"{get_current_datetime()} - Start preprocessing")
    start_t = time.time()

    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        df_features = await loop.run_in_executor(
            executor,
            process_features_parallel,
            df_features,
            os.cpu_count()
        )

    df_concat = pd.concat([df_features, df_labels], axis=1)
    score_counts = df_concat[label].value_counts().sort_index()

    logger.info(f"Preprocessing completed in {time.time() - start_t:.2f}s")
    
    config = {"features": features, "label": label}
    random_10_rows = await save_db(df_concat, db_str, config)
    
    return {"result": random_10_rows, "reduced_score": score_counts.to_dict()}


def process_features_parallel(df: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    chunks = np.array_split(df, n_workers)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        processed = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    return pd.concat(processed).sort_index()


def process_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    cleaned = df_chunk.apply(lambda col: col.str.replace(URL_PATTERN, '', regex=True))
    cleaned = cleaned.apply(lambda col: col.str.replace(EMOJI_PATTERN, '', regex=True))
    cleaned = cleaned.apply(lambda col: col.str.replace(WHITESPACE_PATTERN, ' ', regex=True))
    cleaned = cleaned.apply(lambda col: col.str.strip())
    
    segmented = cleaned.map(word_segmentation)
    
    return segmented


def word_segmentation(text: str) -> str:
    return "".join(rdrsegmenter.word_segment(text))


def get_datasets_from_csa_be(
    client: MongoClient, db_id: str, obj_id_str: str, cols: List[str]
) -> pd.DataFrame:
    db = client.get_database(db_id)
    records_obj = db.get_collection(obj_id_str)
    projection = {col: 1 for col in cols}
    projection.update({"_id": 0})
    docs = list(records_obj.find({}, projection))
    df = pd.DataFrame(docs)
    df = df.fillna("")
    return df


async def save_db(
    df: pd.DataFrame,
    db_id: str,
    config: dict,
) -> List[dict]:
    global client

    _ids = [str(ObjectId()) for _ in range(len(df))]
    df.insert(0, "_id", _ids)

    records = df.to_dict(orient="records")
    total_docs = len(records)

    db = client.get_database(db_id)
    config_coll = db.get_collection(Collections.DATASET_CONFIG)

    if await config_coll.count_documents({}) > 0:
        await config_coll.drop()
        await config_coll.insert_one(
            {"_id": str(ObjectId()), **config, "created_at": get_current_datetime()}
        )

    else:

        await config_coll.replace_one(
            {},
            {
                **config,
                "created_at": get_current_datetime(),
            },
            upsert=True,
        )

    dataset_coll = db.get_collection(Collections.MODEL_DATASET)
    num_docs = await dataset_coll.count_documents({})

    if num_docs > 0:
        await dataset_coll.drop()

    await dataset_coll.insert_many(records)
    
    random_records = await dataset_coll.aggregate([
        {"$sample": {"size": 10}}
    ]).to_list(length=10)

    # random 10 records
    return random_records


async def convert_file_to_df(file: UploadFile) -> pd.DataFrame:
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() == "csv":
        return pd.read_csv(file.file, keep_default_na=False)
    elif file_extension.lower() == "json":
        default = "lines"
        file.file.seek(0)
        first_char = await file.read(1)
        file.file.seek(0)
        if first_char == b"[":
            default = "array"

        if default == "lines":
            return pd.read_json(file.file, lines=True)
        else:
            # print(file.file, type(file.file))
            data = json.load(file.file)
            return pd.DataFrame(data)

    raise Exception(f"Invalid file")


# def translate_text(text, target_language='vi'):
#     translator = Translator()
#     translator.raise_Exception = True
#     if not text:
#         return ""
#     translated_text = translator.translate(text, dest=target_language)
#     return translated_text.text
# df = get_datasets_from_csa_be(client)
# df = preprocess_datasets(df, ["title", "pos_rw", "neg_rw"])


# def get_phobert_tokenizer_config() -> dict:
#     # ["<s>": 0 -> [CLS], "</s>": 2, "<unk>": 3, "<pad>": 1, "<mask>": 64000]
#     special_tokens = tokenizer.all_special_tokens
#     special_ids = tokenizer.all_special_ids

#     return {token: id for token, id in zip(special_tokens, special_ids)}
