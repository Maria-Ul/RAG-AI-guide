# data_caching.py

import json
import os
from datetime import datetime, timedelta

CACHE_DIR = "cache"
CACHE_DURATION = timedelta(hours=12)  # Кэшируем на 12 часов


def load_from_cache(key: str):
    """
    Загружает данные из кэша по ключу, если они актуальны.

    :param key: Ключ для кэшированных данных.
    :return: Кэшированные данные или None, если данные устарели или отсутствуют.
    """
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(cache_file):
        return None

    with open(cache_file, "r") as file:
        data = json.load(file)

    timestamp = datetime.fromisoformat(data["timestamp"])
    if datetime.now() - timestamp > CACHE_DURATION:
        os.remove(cache_file)
        return None

    return data["data"]


def save_to_cache(key: str, data):
    """
    Сохраняет данные в кэш.

    :param key: Ключ для кэширования данных.
    :param data: Данные для сохранения.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")

    try:
        with open(cache_file, "w") as file:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, file)
    except Exception as e:
        print(f"Error: {str(e)}")
