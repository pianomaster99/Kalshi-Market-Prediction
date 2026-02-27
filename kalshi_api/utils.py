import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import base64
import time

load_dotenv()

API_KEY_ID = os.getenv('KALSHI_API_KEY')
PRIVATE_KEY_PATH = os.getenv('KALSHI_PRIVATE_KEY_PATH')

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

private_key = load_private_key(PRIVATE_KEY_PATH)

def sign_pss_text(text: str, private_key=private_key) -> str:
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def create_signature(timestamp, method, path, private_key=private_key):
    path_without_query = path.split('?')[0]

    message = f"{timestamp}{method}{path_without_query}"

    return sign_pss_text(message, private_key)

def create_headers(method: str, path: str, private_key=private_key) -> dict:
    timestamp = str(int(time.time() * 1000))
    signature = create_signature(timestamp, method, path, private_key)
    return {
         "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }