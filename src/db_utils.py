"""
DynamoDB Local yardımcı modülü.

Bu modül SADECE yerel DynamoDB (http://localhost:8000) ile çalışır.
AWS bulut DynamoDB'ye bağlanmaz.

DynamoDB Local'i başlatmak için:
    docker-compose up -d

Bağlantı ayarları:
    endpoint_url : http://localhost:8000
    region_name  : local
    credentials  : dummy (yerel için doğrulama gerekmez)
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import boto3
from botocore.exceptions import EndpointResolutionError, NoRegionError

# ---------------------------------------------------------------------------
# Bağlantı sabitleri – AWS bulut bağlantısı YOKTUR
# ---------------------------------------------------------------------------
DYNAMODB_ENDPOINT = "http://localhost:8000"
REGION = "local"
_DUMMY_KEY = "local"

# Tablo adları
TABLE_SONG_ANALYSES = "song_analyses"
TABLE_GEMINI_ANALYSES = "gemini_analyses"


# ---------------------------------------------------------------------------
# İç yardımcılar
# ---------------------------------------------------------------------------

def _to_decimal(value):
    """Python float/int değerini DynamoDB uyumlu Decimal'e çevirir."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal("0")


def _from_decimal(obj):
    """DynamoDB'den dönen Decimal değerlerini float'a çevirir."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(i) for i in obj]
    return obj


def _get_resource():
    """
    DynamoDB Local kaynağı döndürür.
    Yerel sunucuya ulaşılamazsa None döner – AWS'e asla bağlanmaz.
    """
    return boto3.resource(
        "dynamodb",
        endpoint_url=DYNAMODB_ENDPOINT,
        region_name=REGION,
        aws_access_key_id=_DUMMY_KEY,
        aws_secret_access_key=_DUMMY_KEY,
    )


def _create_tables_if_needed(dynamodb):
    """Gerekli tabloları yoksa oluşturur."""
    existing = {t.name for t in dynamodb.tables.all()}

    if TABLE_SONG_ANALYSES not in existing:
        dynamodb.create_table(
            TableName=TABLE_SONG_ANALYSES,
            KeySchema=[{"AttributeName": "analysis_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "analysis_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    if TABLE_GEMINI_ANALYSES not in existing:
        dynamodb.create_table(
            TableName=TABLE_GEMINI_ANALYSES,
            KeySchema=[{"AttributeName": "analysis_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "analysis_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",
        )


def get_db():
    """
    DynamoDB Local bağlantısı döndürür.

    Yerel sunucu (localhost:8000) çalışmıyorsa None döner.
    AWS bulut DynamoDB kullanılmaz.
    """
    try:
        db = _get_resource()
        _create_tables_if_needed(db)
        return db
    except Exception:
        return None


def is_local_db_available():
    """DynamoDB Local'in çalışıp çalışmadığını kontrol eder."""
    try:
        db = _get_resource()
        list(db.tables.all())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Yazma işlemleri
# ---------------------------------------------------------------------------

def save_song_analysis(
    song_name: str,
    model_name: str,
    model_valence: float,
    model_arousal: float,
    human_valence_mean: float,
    human_arousal_mean: float,
    human_valence_std: float,
    human_arousal_std: float,
    delta_v: float,
    delta_a: float,
    n_responses: int,
) -> bool:
    """
    Model karşılaştırma sonucunu DynamoDB Local'e kaydeder.

    Returns:
        True  – kayıt başarılı
        False – DynamoDB Local çalışmıyor veya hata oluştu
    """
    db = get_db()
    if db is None:
        return False

    try:
        table = db.Table(TABLE_SONG_ANALYSES)
        table.put_item(
            Item={
                "analysis_id": str(uuid.uuid4()),
                "song_name": song_name,
                "model_name": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_valence": _to_decimal(model_valence),
                "model_arousal": _to_decimal(model_arousal),
                "human_valence_mean": _to_decimal(human_valence_mean),
                "human_arousal_mean": _to_decimal(human_arousal_mean),
                "human_valence_std": _to_decimal(human_valence_std),
                "human_arousal_std": _to_decimal(human_arousal_std),
                "delta_v": _to_decimal(delta_v),
                "delta_a": _to_decimal(delta_a),
                "n_responses": int(n_responses),
            }
        )
        return True
    except Exception:
        return False


def save_gemini_analysis(
    song_name: str,
    gemini_valence: float,
    gemini_arousal: float,
    gemini_emotion: str,
    human_valence_mean: float,
    human_arousal_mean: float,
    delta_v: float,
    delta_a: float,
) -> bool:
    """
    Gemini AI analiz sonucunu DynamoDB Local'e kaydeder.

    Returns:
        True  – kayıt başarılı
        False – DynamoDB Local çalışmıyor veya hata oluştu
    """
    db = get_db()
    if db is None:
        return False

    try:
        table = db.Table(TABLE_GEMINI_ANALYSES)
        table.put_item(
            Item={
                "analysis_id": str(uuid.uuid4()),
                "song_name": song_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gemini_valence": _to_decimal(gemini_valence),
                "gemini_arousal": _to_decimal(gemini_arousal),
                "gemini_emotion": gemini_emotion or "",
                "human_valence_mean": _to_decimal(human_valence_mean),
                "human_arousal_mean": _to_decimal(human_arousal_mean),
                "delta_v": _to_decimal(delta_v),
                "delta_a": _to_decimal(delta_a),
            }
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Okuma işlemleri
# ---------------------------------------------------------------------------

def get_all_song_analyses() -> list:
    """
    Tüm model analizi kayıtlarını DynamoDB Local'den döndürür.
    Decimal değerler float'a çevrilmiş şekilde döner.
    """
    db = get_db()
    if db is None:
        return []

    try:
        table = db.Table(TABLE_SONG_ANALYSES)
        response = table.scan()
        items = response.get("Items", [])

        # Sayfalama (1 MB sınırını aşan veri setleri için)
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        return [_from_decimal(item) for item in items]
    except Exception:
        return []


def get_all_gemini_analyses() -> list:
    """
    Tüm Gemini analizi kayıtlarını DynamoDB Local'den döndürür.
    Decimal değerler float'a çevrilmiş şekilde döner.
    """
    db = get_db()
    if db is None:
        return []

    try:
        table = db.Table(TABLE_GEMINI_ANALYSES)
        response = table.scan()
        items = response.get("Items", [])

        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        return [_from_decimal(item) for item in items]
    except Exception:
        return []


def delete_song_analysis(analysis_id: str) -> bool:
    """Belirtilen model analizi kaydını siler."""
    db = get_db()
    if db is None:
        return False
    try:
        db.Table(TABLE_SONG_ANALYSES).delete_item(
            Key={"analysis_id": analysis_id}
        )
        return True
    except Exception:
        return False


def delete_gemini_analysis(analysis_id: str) -> bool:
    """Belirtilen Gemini analizi kaydını siler."""
    db = get_db()
    if db is None:
        return False
    try:
        db.Table(TABLE_GEMINI_ANALYSES).delete_item(
            Key={"analysis_id": analysis_id}
        )
        return True
    except Exception:
        return False
