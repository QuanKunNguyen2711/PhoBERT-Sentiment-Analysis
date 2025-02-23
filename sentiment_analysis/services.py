import asyncio
import copy
import os
import time
from typing import List
from bson import ObjectId
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
import logging
import torch.nn as nn
import torch
import pandas as pd
# from data_pipeline.services import get_train_val_test_dataloader
from data_pipeline.services import get_optimized_dataloaders
import torch.cuda
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from app.common.db_collections import Collections
from app.common.db_connector import client
from app.common.utils import generate_model_id, get_current_datetime
from app.common.websocket import WebsocketEventResult, ws_manager
from data_pipeline.services import tokenizer

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL = "vinai/phobert-base"

class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 64, dropout: float = 0.3):
        super(SentimentAnalysisModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(MODEL)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        first_cls_token = outputs[0][:, 0, :]
        return self.classifier(first_cls_token)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0005):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        return False
    

class CheckpointSaver:
    def __init__(self, db_str: str):
        save_dir = f"{os.environ.get("ROOT_PATH")}/checkpoints/{db_str}"
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir
        self.best_val_loss = float("inf")

    def save_checkpoint(self, model, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(
                model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                os.path.join(self.save_path, "best_model.pth")
            )

    def load_best_model(self, model):
        state_dict = torch.load(os.path.join(self.save_path, "best_model.pth"))
        model.load_state_dict(state_dict)
        return model

async def fine_tuning_model(
    df: pd.DataFrame,
    cols: List[str],
    training_epoch_coll,
    name: str,
    description: str,
    db_str: str,
    cur_user_id: str,
    hidden_size: int = 64,
    batch_size: int = 128,
    num_epochs: int = 50,
):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        num_classes = len(df[cols[-1]].unique())
        df[cols[-1]] = df[cols[-1]] - df[cols[-1]].min()
        
        model = SentimentAnalysisModel(num_classes=num_classes, hidden_size=hidden_size)
        model_id_str = generate_model_id(name)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            eta_min=1e-7
        )

        scaler = torch.amp.GradScaler('cuda')
        
        # Data loading
        train_loader, val_loader, test_loader, class_weights = get_optimized_dataloaders(
            df, cols, batch_size
        )
        
        # Loss function
        class_weights = class_weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training components
        early_stopping = EarlyStopping(tolerance=2)
        checkpoint_saver = CheckpointSaver(db_str=db_str)

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"{get_current_datetime()} - Start epoch {epoch+1}")
            
            # Training
            train_loss = await async_train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )
            
            # Validation
            val_loss = await async_validate_epoch(
                model, val_loader, loss_fn, device
            )
            
            # Update learning rate
            scheduler.step()
            
            # Logging and checkpointing
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            checkpoint_saver.save_checkpoint(model, val_loss)
            
            # Early stopping check
            if early_stopping(val_loss):                    
                logger.info(f"Early stopping at epoch {epoch+1}")
                await ws_manager.send_ws({"event": WebsocketEventResult.EARLY_STOPPING, "early_stopping": f"{get_current_datetime()} - Early stopping at {epoch+1}/{num_epochs}"}, cur_user_id)
                
                break
            
            # WebSocket updates
            epoch_info = {
                "epoch": f"{epoch+1}/{num_epochs}",
                "model_id": model_id_str,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "created_at": get_current_datetime(),
            }
            await ws_manager.send_ws({**epoch_info, "event": WebsocketEventResult.TRAINING_EPOCH}, cur_user_id)
            await training_epoch_coll.insert_one({**epoch_info, "_id": str(ObjectId())})
            
        # Final evaluation
        metrics = await async_evaluate_model(model, test_loader, device)
        await ws_manager.send_ws({**metrics, "event": WebsocketEventResult.EVALUATION_METRIC}, cur_user_id)
        await save_model_db(model, metrics, db_str, name, description)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")

async def async_train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            inputs = batch["input_ids"].to(device, non_blocking=True)
            masks = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            outputs = model(inputs, masks)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

async def async_validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            masks = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            outputs = model(inputs, masks)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

async def async_evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            masks = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()
            
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return {
        "precision": precision_score(all_labels, all_preds, average="weighted"),
        "recall": recall_score(all_labels, all_preds, average="weighted"),
        "f1_score": f1_score(all_labels, all_preds, average="weighted"),
        "accuracy": accuracy_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


loaded_models = {}

def infer_prediction(text: str, model_id: str, db_str: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model_key = (model_id, db_str)
    if model_key not in loaded_models:
        model_path = os.path.join(os.environ.get("ROOT_PATH", ""), "models", db_str, f"{model_id}.pth")
        
        model = torch.load(model_path, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        model = model.to(device)
        model.eval()
        
        if device.type == 'cuda':
            model = model.half()
        
        loaded_models[model_key] = model
    else:
        model = loaded_models[model_key]

    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            # Use autocast for mixed precision on CUDA
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask)
        else:
            outputs = model(input_ids, attention_mask)
        
        prob = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(prob, dim=1).item() + 1

    return predicted_class


async def save_model_db(
    model,
    evaluation_metric: dict,
    db_str: str,
    model_name: str,
    model_description: str,
):
    save_dir = f"{os.environ.get("ROOT_PATH")}/models/{db_str}"
    os.makedirs(save_dir, exist_ok=True)

    model_id_str = generate_model_id(model_name)

    torch.save(model, os.path.join(save_dir, f"{model_id_str}.pth"))

    db = client.get_database(db_str)
    model_col = db.get_collection(Collections.SENTIMENT_MODEL)
    if await model_col.count_documents({}) > 0:
        await model_col.insert_one(
            {
                "_id": str(ObjectId()),
                "name": model_name,
                "description": model_description,
                "model_id": model_id_str,
                **evaluation_metric,
                "created_at": get_current_datetime(),
            }
        )

    else:
        await model_col.replace_one(
            {},
            {
                "name": model_name,
                "description": model_description,
                "model_id": model_id_str,
                **evaluation_metric,
                "created_at": get_current_datetime(),
            },
            upsert=True,
        )
