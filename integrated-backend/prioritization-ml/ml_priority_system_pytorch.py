#!/usr/bin/env python3
"""
ml_priority_system_pytorch.py

PyTorch priority classifier + FastAPI + RabbitMQ worker
Now includes:
 - JSON structured logging (suitable for Logstash)
 - Prometheus metrics (exposed at /metrics)
 - RabbitMQ TTL retry queues + dead-letter queue for retry logic

Usage:
 - Train:
     python ml_priority_system_pytorch.py --train data.csv --epochs 5 --save_dir models/
 - Start API (loads model at startup):
     python ml_priority_system_pytorch.py --serve --host 0.0.0.0 --port 8000
 - Start worker:
     python ml_priority_system_pytorch.py --worker

Requirements:
 pip install fastapi uvicorn[standard] pika torch pandas numpy joblib PyPDF2 python-multipart scikit-learn prometheus_client
"""
import os
import io
import json
import argparse
import base64
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import logging
import socket
import traceback

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pika
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split

# prometheus
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_client import make_asgi_app

# -------------------------
# Config / Defaults
# -------------------------
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_DIR.mkdir(exist_ok=True)
VOCAB_FILENAME = "vocab.json"
MODEL_FILENAME = "priority_model.pt"
CONFIG_FILENAME = "model_config.json"

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
INPUT_QUEUE = os.getenv("INPUT_QUEUE", "priority_queue")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE", "cases_out")
DEAD_QUEUE = os.getenv("DEAD_QUEUE", "cases_dead")

# retry queue definitions: (name_suffix, ttl_ms)
RETRY_STAGES = [("retry_1", 5000), ("retry_2", 30000), ("retry_3", 120000)]
MAX_RETRIES = len(RETRY_STAGES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global objects loaded at startup (FastAPI)
GLOBAL_MODEL: Optional[nn.Module] = None
GLOBAL_VOCAB: Optional[Dict[str, Any]] = None
GLOBAL_CONFIG: Optional[Dict[str, Any]] = None
GLOBAL_MODEL_DIR = DEFAULT_MODEL_DIR

# -------------------------
# Logging (JSON) for Logstash
# -------------------------
class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        msg = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # include extra fields if provided in record.__dict__
        extras = {k: v for k, v in record.__dict__.items() if k not in ("name","msg","args","levelname","levelno","pathname","filename","module","exc_info","exc_text","stack_info","lineno","funcName","created","msecs","relativeCreated","thread","threadName","processName","process")}
        if extras:
            # try to make JSON-serializable
            try:
                json.dumps(extras)
                msg["extra"] = extras
            except Exception:
                # fallback: include keys only
                msg["extra_keys"] = list(extras.keys())
        if record.exc_info:
            msg["exc"] = self.formatException(record.exc_info)
        return json.dumps(msg)

def setup_logging(level=logging.INFO, logstash_host=None, logstash_port=None):
    logger = logging.getLogger("priority")
    logger.setLevel(level)
    handler = logging.StreamHandler()  # stdout; in k8s this goes to container logs
    handler.setFormatter(JsonLogFormatter())
    logger.handlers = [handler]
    # Optionally add TCP socket handler to ship logs directly to logstash if host/port provided
    if logstash_host and logstash_port:
        try:
            sock_handler = logging.handlers.SocketHandler(logstash_host, int(logstash_port))
            sock_handler.setFormatter(JsonLogFormatter())
            logger.addHandler(sock_handler)
        except Exception:
            logger.warning("Failed to add socket handler to Logstash", exc_info=True)
    return logger

LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", None)
LOGSTASH_PORT = os.getenv("LOGSTASH_PORT", None)
logger = setup_logging(level=logging.INFO, logstash_host=LOGSTASH_HOST, logstash_port=LOGSTASH_PORT)

# -------------------------
# Prometheus metrics
# -------------------------
REGISTRY = CollectorRegistry()
cases_processed = Counter("cases_processed_total", "Total number of cases processed (including success/failure)", registry=REGISTRY)
cases_success = Counter("cases_success_total", "Total number of successful predictions", registry=REGISTRY)
cases_failed = Counter("cases_failed_total", "Total number of failed attempts", registry=REGISTRY)
cases_retried = Counter("cases_retried_total", "Total number of retried messages", registry=REGISTRY)
cases_dead = Counter("cases_dead_total", "Total number of messages sent to dead queue", registry=REGISTRY)
processing_time = Histogram("case_processing_seconds", "Case processing time in seconds", registry=REGISTRY)
# gauge for queue lengths (we will set values periodically)
queue_messages = Gauge("rabbitmq_queue_messages", "Messages in RabbitMQ queues (label=queue)", ["queue"], registry=REGISTRY)

# -------------------------
# Utilities: PDF extraction
# -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""

# -------------------------
# Simple tokenizer & vocab
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    for ch in "\n\t\r,;:.()[]{}!?/\\|-*_\"'`<>":
        text = text.replace(ch, " ")
    tokens = [tok for tok in text.split() if tok]
    return tokens

def build_vocab(texts: List[str], min_freq: int = 2, max_size: Optional[int] = 30000):
    freq = {}
    for t in texts:
        for tok in simple_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    tokens = [tok for tok, c in items if c >= min_freq]
    if max_size:
        tokens = tokens[:max_size]
    stoi = {"<pad>": 0, "<unk>": 1}
    for i, tok in enumerate(tokens, start=len(stoi)):
        stoi[tok] = i
    itos = {i: s for s, i in stoi.items()}
    return {"stoi": stoi, "itos": itos}

def text_to_indices(text: str, stoi: Dict[str,int], max_len: int):
    toks = simple_tokenize(text)
    idxs = [stoi.get(t, stoi["<unk>"]) for t in toks][:max_len]
    if len(idxs) < max_len:
        idxs = idxs + [stoi["<pad>"]] * (max_len - len(idxs))
    return idxs

# -------------------------
# Dataset & Dataloader
# -------------------------
class SymptomDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], stoi: Dict[str,int], max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        x = torch.tensor(text_to_indices(txt, self.stoi, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # labels: 1/2/3 -> 0/1/2
        return x, y

# -------------------------
# PyTorch model
# -------------------------
class PriorityModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, n_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        fd = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(fd, max(32, fd//2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, fd//2), num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out_t = out.permute(0, 2, 1)
        pooled = self.pool(out_t).squeeze(-1)
        logits = self.fc(pooled)
        return logits

# -------------------------
# Training & evaluation
# -------------------------

def train_pytorch(csv_path: str,
                  save_dir: str = str(DEFAULT_MODEL_DIR),
                  epochs: int = 5,
                  batch_size: int = 32,
                  lr: float = 1e-3,
                  max_len: int = 128,
                  embed_dim: int = 128,
                  hidden_dim: int = 128,
                  min_freq: int = 2):
    df = pd.read_csv(csv_path)
    if "symptoms_text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: symptoms_text and label (values 1/2/3)")

    texts = df["symptoms_text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # build vocab
    vocab = build_vocab(texts, min_freq=min_freq)
    stoi = vocab["stoi"]

    # -------------------------
    # Robust stratified train/val split
    # -------------------------
    n_samples = len(labels)
    classes = set(labels)
    n_classes = len(classes)

    desired_frac = 0.12
    abs_test = max(int(round(desired_frac * n_samples)), n_classes)

    if abs_test >= n_samples:
        if n_samples > n_classes:
            split_idx = max(1, n_samples - n_classes)
            X_train, X_val = texts[:split_idx], texts[split_idx:]
            y_train, y_val = labels[:split_idx], labels[split_idx:]
        elif n_samples > 1:
            X_train, X_val, y_train, y_val = texts[:-1], texts[-1:], labels[:-1], labels[-1:]
        else:
            X_train, X_val, y_train, y_val = texts, [], labels, []
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=abs_test, random_state=42, stratify=labels
        )

    train_ds = SymptomDataset(X_train, y_train, stoi, max_len=max_len)
    val_ds = SymptomDataset(X_val, y_val, stoi, max_len=max_len) if len(X_val) > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    model = PriorityModel(vocab_size=len(stoi), embed_dim=embed_dim, hidden_dim=hidden_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total if total>0 else 0.0
        train_acc = correct / total if total>0 else 0.0

        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == yb).sum().item()
                    v_total += xb.size(0)
            val_acc = v_correct / v_total if v_total>0 else 0.0

        logger.info(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            model_file = save_path / MODEL_FILENAME
            vocab_file = save_path / VOCAB_FILENAME
            config_file = save_path / CONFIG_FILENAME
            torch.save(model.state_dict(), str(model_file))
            with open(vocab_file, "w") as f:
                json.dump(vocab, f)
            config = {"max_len": max_len, "embed_dim": embed_dim, "hidden_dim": hidden_dim}
            with open(config_file, "w") as f:
                json.dump(config, f)
            logger.info(f"Saved best model (val_acc={best_val_acc:.4f}) to {save_path}")
        elif val_loader is None:
            model_file = save_path / MODEL_FILENAME
            vocab_file = save_path / VOCAB_FILENAME
            config_file = save_path / CONFIG_FILENAME
            torch.save(model.state_dict(), str(model_file))
            with open(vocab_file, "w") as f:
                json.dump(vocab, f)
            config = {"max_len": max_len, "embed_dim": embed_dim, "hidden_dim": hidden_dim}
            with open(config_file, "w") as f:
                json.dump(config, f)
            logger.info(f"Saved model (no validation available) to {save_path}")

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc}

# -------------------------
# Helpers: load model and predict
# -------------------------
def load_artifacts(model_dir: str = str(DEFAULT_MODEL_DIR)):
    model_dir = Path(model_dir)
    vocab_path = model_dir / VOCAB_FILENAME
    model_path = model_dir / MODEL_FILENAME
    config_path = model_dir / CONFIG_FILENAME

    if not vocab_path.exists() or not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}. Expected {vocab_path}, {model_path}, {config_path}")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)

    model = PriorityModel(vocab_size=len(vocab["stoi"]), embed_dim=config.get("embed_dim", 128), hidden_dim=config.get("hidden_dim", 128))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, vocab, config

def predict_text(model: nn.Module, vocab: Dict[str, Any], config: Dict[str, Any], text: str):
    stoi = vocab["stoi"]
    max_len = int(config.get("max_len", 128))
    idxs = text_to_indices(text, stoi, max_len)
    xb = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=-1).cpu().item()) + 1  # back to 1/2/3
    return {"priority": pred, "probs": probs.tolist()}

# -------------------------
# FastAPI app + metrics endpoint
# -------------------------
app = FastAPI(title="PriorityClassifier")

# mount Prometheus ASGI app at /metrics
metrics_app = make_asgi_app(registry=REGISTRY)
app.mount("/metrics", metrics_app)

class EnqueueResponse(BaseModel):
    status: str
    message_id: Optional[str] = None
    detail: Optional[str] = None

class PredictRequest(BaseModel):
    symptoms_text: str
    history_pdf_base64: Optional[str] = None
    case_id: Optional[str] = None

@app.on_event("startup")
def load_model_on_startup():
    global GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG, GLOBAL_MODEL_DIR
    try:
        model_dir = os.getenv("MODEL_DIR", str(GLOBAL_MODEL_DIR))
        logger.info("startup.loading", extra={"model_dir": model_dir})
        GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG = load_artifacts(model_dir)
        GLOBAL_MODEL_DIR = Path(model_dir)
        logger.info("startup.model_loaded", extra={"device": str(DEVICE)})
    except Exception as e:
        logger.warning("startup.model_load_failed", extra={"error": str(e)})
        GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG = None, None, None

@app.on_event("shutdown")
def shutdown_event():
    logger.info("shutdown")

@app.post("/enqueue", response_model=EnqueueResponse)
async def enqueue_case(symptoms_text: str = Form(...), history_pdf: Optional[UploadFile] = File(None), case_id: Optional[str] = Form(None)):
    payload = {"case_id": case_id, "symptoms_text": symptoms_text, "retries": 0}
    if history_pdf is not None:
        data = await history_pdf.read()
        b64 = base64.b64encode(data).decode("utf-8")
        payload["history_pdf_base64"] = b64

    try:
        conn = pika.URLParameters(RABBITMQ_URL)
        connection = pika.BlockingConnection(conn)
        channel = connection.channel()
        # ensure all queues exist
        declare_queues(channel)
        channel.basic_publish(exchange="", routing_key=INPUT_QUEUE, body=json.dumps(payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
        connection.close()
        logger.info("case_enqueued", extra={"case_id": case_id})
        return {"status": "ok", "message_id": case_id}
    except Exception as e:
        logger.error("enqueue_failed", extra={"case_id": case_id, "error": str(e), "trace": traceback.format_exc()})
        return {"status": "error", "message_id": None, "detail": str(e)}

@app.post("/predict")
async def predict_sync(req: PredictRequest):
    global GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG, GLOBAL_MODEL_DIR
    if GLOBAL_MODEL is None:
        # fallback attempt to load artifacts
        try:
            GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG = load_artifacts(str(GLOBAL_MODEL_DIR))
        except Exception as e:
            logger.error("predict_no_model", extra={"error": str(e)})
            return JSONResponse(status_code=500, content={"error": "model not available", "detail": str(e)})

    model = GLOBAL_MODEL
    vocab = GLOBAL_VOCAB
    config = GLOBAL_CONFIG

    text = req.symptoms_text or ""
    if req.history_pdf_base64:
        try:
            pdf_bytes = base64.b64decode(req.history_pdf_base64)
            hist_text = extract_text_from_pdf_bytes(pdf_bytes)
            text = text + "\n" + hist_text
        except Exception:
            pass

    start = time.time()
    res = predict_text(model, vocab, config, text)
    processing_time.observe(time.time() - start)
    cases_processed.inc()
    cases_success.inc()
    logger.info("predict_sync", extra={"case_id": req.case_id, "priority": res["priority"]})
    return {"case_id": req.case_id, "priority": res["priority"], "probs": res["probs"]}

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}

# -------------------------
# RabbitMQ helpers (retry queues, DLQ)
# -------------------------
DLX_EXCHANGE = os.getenv("RABBITMQ_DLX", "hospital_dlx")

def declare_queues(channel):
    # Main input/output/dead queues
    queue_args = {
        "x-dead-letter-exchange": DLX_EXCHANGE,
        "x-message-ttl": 3600000  # 1 hour
    }
    
    # Ensure DLX exists first (idempotent)
    channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type='topic', durable=True)
    
    channel.queue_declare(queue=INPUT_QUEUE, durable=True, arguments=queue_args)
    channel.queue_declare(queue=OUTPUT_QUEUE, durable=True, arguments=queue_args)
    channel.queue_declare(queue=DEAD_QUEUE, durable=True)

    # Retry queues: each has TTL and dead-letters back to INPUT_QUEUE
    for name_suffix, ttl in RETRY_STAGES:
        qname = f"{INPUT_QUEUE}_{name_suffix}"
        args = {
            "x-dead-letter-exchange": "",  # default exchange
            "x-dead-letter-routing-key": INPUT_QUEUE,
            "x-message-ttl": ttl
        }
        channel.queue_declare(queue=qname, durable=True, arguments=args)
    logger.info("queues_declared", extra={"input": INPUT_QUEUE, "output": OUTPUT_QUEUE, "dead": DEAD_QUEUE, "retries": [s for s,_ in RETRY_STAGES]})

def get_queue_message_count(channel, qname):
    # passive declare to get message count
    try:
        res = channel.queue_declare(queue=qname, passive=True)
        return res.method.message_count
    except Exception:
        return None

# -------------------------
# RabbitMQ worker with retry/dlq logic
# -------------------------
def run_worker(model_dir: str = str(DEFAULT_MODEL_DIR)):
    global GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG, GLOBAL_MODEL_DIR
    logger.info("worker.starting")
    # Try to reuse global model if present; otherwise load
    if GLOBAL_MODEL is None:
        try:
            GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG = load_artifacts(model_dir)
            GLOBAL_MODEL_DIR = Path(model_dir)
            logger.info("worker.model_loaded")
        except Exception as e:
            logger.warning("worker.model_load_failed", extra={"error": str(e)})

    conn_params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(conn_params)
    channel = connection.channel()

    # declare queues (main + retry + dead)
    declare_queues(channel)

    # update queue metrics initially
    try:
        for q in [INPUT_QUEUE, OUTPUT_QUEUE, DEAD_QUEUE] + [f"{INPUT_QUEUE}_{s}" for s,_ in RETRY_STAGES]:
            c = get_queue_message_count(channel, q)
            if c is not None:
                queue_messages.labels(queue=q).set(c)
    except Exception:
        pass

    def callback(ch, method, properties, body):
        start = time.time()
        cases_processed.inc()
        try:
            payload = json.loads(body.decode("utf-8"))
            case_id = payload.get("case_id")
            symptoms_text = payload.get("symptoms_text", "")
            retries = int(payload.get("retries", 0))
            hist_b64 = payload.get("history_pdf_base64")
            if hist_b64:
                try:
                    pdf_bytes = base64.b64decode(hist_b64)
                    hist_text = extract_text_from_pdf_bytes(pdf_bytes)
                    symptoms_text = symptoms_text + "\n" + hist_text
                except Exception:
                    pass

            # Use rule-based priority calculation
            try:
                from priority_rules import calculate_priority_from_symptoms
                rule_priority = calculate_priority_from_symptoms(symptoms_text)
                
                # If ML model is available, use it but validate against rules
                if GLOBAL_MODEL is not None and GLOBAL_VOCAB is not None and GLOBAL_CONFIG is not None:
                    try:
                        with processing_time.time():
                            result = predict_text(GLOBAL_MODEL, GLOBAL_VOCAB, GLOBAL_CONFIG, symptoms_text)
                        # Use rule-based priority for now since ML model is not well-trained
                        # In production, you could use ML priority if confidence is high
                        final_priority = rule_priority
                        out_payload = {"case_id": case_id, "priority": final_priority, "probs": result["probs"]}
                        channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=json.dumps(out_payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        cases_success.inc()
                        logger.info("worker.processed_success", extra={"case_id": case_id, "priority": final_priority, "ml_priority": result["priority"], "rule_priority": rule_priority, "retries": retries, "duration_s": time.time()-start})
                    except Exception as e:
                        # error during prediction -> use rule-based priority
                        logger.warning("worker.ml_error_using_rules", extra={"case_id": case_id, "error": str(e)})
                        out_payload = {"case_id": case_id, "priority": rule_priority, "probs": [0.33, 0.33, 0.34]}
                        channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=json.dumps(out_payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        cases_success.inc()
                else:
                    # model not available -> use rule-based priority
                    logger.info("worker.using_rule_based_priority", extra={"case_id": case_id, "priority": rule_priority})
                    out_payload = {"case_id": case_id, "priority": rule_priority, "probs": [0.33, 0.33, 0.34]}
                    channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=json.dumps(out_payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    cases_success.inc()
            except Exception as e:
                # fallback to medium priority if rules fail
                logger.error("worker.rule_based_error", extra={"case_id": case_id, "error": str(e)})
                out_payload = {"case_id": case_id, "priority": 2, "probs": [0.33, 0.33, 0.34]}
                channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=json.dumps(out_payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
                ch.basic_ack(delivery_tag=method.delivery_tag)
                cases_failed.inc()
        except Exception as e:
            # unexpected error parsing message -> ack to avoid infinite loop (or send to dead)
            logger.error("worker.unexpected_error", extra={"error": str(e), "trace": traceback.format_exc()})
            try:
                # try to push to dead queue
                channel.basic_publish(exchange="", routing_key=DEAD_QUEUE, body=body, properties=pika.BasicProperties(delivery_mode=2))
                cases_dead.inc()
                logger.info("worker.sent_to_dead_on_unexpected")
            except Exception:
                logger.exception("worker.failed_sending_to_dead")
            ch.basic_ack(delivery_tag=method.delivery_tag)

        # update queue metrics periodically (cheap)
        try:
            c = get_queue_message_count(channel, INPUT_QUEUE)
            if c is not None:
                queue_messages.labels(queue=INPUT_QUEUE).set(c)
        except Exception:
            pass

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)
    logger.info("worker.consuming", extra={"queue": INPUT_QUEUE})
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()

def handle_retry_or_dead(channel, payload):
    """
    If payload.retries < MAX_RETRIES: send to next retry queue (with incremented retries)
    Else: send to DEAD_QUEUE
    """
    retries = int(payload.get("retries", 0))
    case_id = payload.get("case_id")
    if retries < MAX_RETRIES:
        # publish to the retry queue corresponding to this retry count (0 -> retry_1)
        qname, ttl = RETRY_STAGES[retries]
        retry_qname = f"{INPUT_QUEUE}_{qname}"
        payload_copy = dict(payload)
        payload_copy["retries"] = retries + 1
        try:
            channel.basic_publish(exchange="", routing_key=retry_qname, body=json.dumps(payload_copy).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
            cases_retried.inc()
            logger.info("worker.sent_to_retry", extra={"case_id": case_id, "retry_queue": retry_qname, "new_retries": payload_copy["retries"], "ttl_ms": ttl})
        except Exception as e:
            # fallback: send directly to dead queue
            try:
                channel.basic_publish(exchange="", routing_key=DEAD_QUEUE, body=json.dumps(payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
                cases_dead.inc()
                logger.error("worker.retry_failed_sent_to_dead", extra={"case_id": case_id, "error": str(e)})
            except Exception:
                logger.exception("worker.retry_failed_and_dead_failed")
    else:
        # send to dead queue
        try:
            channel.basic_publish(exchange="", routing_key=DEAD_QUEUE, body=json.dumps(payload).encode("utf-8"), properties=pika.BasicProperties(delivery_mode=2))
            cases_dead.inc()
            logger.info("worker.sent_to_dead", extra={"case_id": case_id, "retries": retries})
        except Exception:
            logger.exception("worker.failed_sending_to_dead")

# -------------------------
# CLI
# -------------------------
def cli():
    parser = argparse.ArgumentParser(description="PyTorch Priority Classifier with FastAPI + RabbitMQ + metrics + retry")
    parser.add_argument("--train", type=str, help="Path to training CSV")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker", action="store_true", help="Run RabbitMQ worker")
    args = parser.parse_args()

    if args.train:
        logger.info("cli.train_start", extra={"csv": args.train})
        train_pytorch(args.train, save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, min_freq=args.min_freq)
        logger.info("cli.train_finished")
    elif args.serve:
        logger.info(f"Starting FastAPI server on {args.host}:{args.port} (device={DEVICE})")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.worker:
        run_worker()
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
