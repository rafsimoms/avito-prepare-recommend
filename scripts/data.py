"""Чтение Parquet и подготовка корпуса объявлений в локальной SQLite-базе."""
from itertools import islice
from contextlib import closing
from pathlib import Path
import os
import re
import sqlite3
import time

import pandas as pd
import pyarrow.parquet as pq


def log(message):
    """Показать текущий этап обработки."""
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def clipped_words(value, limit):
    # Читаем только первые limit слов, не создавая список всего описания.
    if value is None or pd.isna(value):
        return ''
    return ' '.join(match.group() for match in islice(re.finditer(r'\S+', str(value)), max(0, limit)))


def read_small_table(path, required):
    """Загрузить только необходимые колонки запросов или взаимодействий."""
    parquet = pq.ParquetFile(path)
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'{path}: отсутствуют колонки {sorted(missing)}')
    frame = parquet.read(columns=required, use_threads=False).to_pandas(types_mapper=pd.ArrowDtype)
    if frame.empty:
        raise ValueError(f'{path}: пустая таблица')
    return frame


class ItemCorpus:
    """Отдавать метаданные и тексты в одном порядке: по item_id."""
    def __init__(self, path):
        self.path = str(Path(path).resolve())

    def metadata(self):
        with closing(sqlite3.connect(self.path)) as db:
            return pd.read_sql_query(
                'SELECT item_id, item_location_id, item_category_id, '
                'item_rating_reviews_count, item_rating FROM items ORDER BY item_id', db)

    def documents(self, kind, title_repeat=3):
        # Курсор читает тексты по одному; порядок совпадает с metadata().
        with closing(sqlite3.connect(self.path)) as db:
            for title, params, body in db.execute('SELECT title, params, body FROM items ORDER BY item_id'):
                if kind == 'bm25':
                    yield (title + ' ') * title_repeat + params + ' ' + body
                else:
                    yield title + ' ' + params


def build_corpus(cfg, destination):
    """Подготовить весь benchmark-корпус, обрезав длинные поля по конфигу."""
    # Локальный импорт нужен из-за общей нормализации категориальных признаков.
    from retrievers import category
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.building.sqlite')
    temporary.unlink(missing_ok=True)
    required = ['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw',
                'item_location_id', 'item_category_id']
    optional = ['item_rating_reviews_count', 'item_rating']
    batch_size = int(cfg.get('memory', {}).get('parquet_batch_rows', 256))
    if batch_size < 1:
        raise ValueError('memory.parquet_batch_rows должен быть положительным')
    seen = set()
    processed = 0
    with closing(sqlite3.connect(temporary)) as db:
        db.execute('PRAGMA cache_size=-16384')  # Кеш страниц SQLite: 16 MiB.
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE items (item_id TEXT PRIMARY KEY, title TEXT, params TEXT, body TEXT, '
                   'item_location_id TEXT, item_category_id TEXT, item_rating_reviews_count REAL, item_rating REAL)')
        parquet = pq.ParquetFile(cfg['paths']['items'])
        missing = set(required) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f'benchmark_items: отсутствуют колонки {sorted(missing)}')
        cols = required + [c for c in optional if c in parquet.schema_arrow.names]
        log(f'Корпус: чтение benchmark_items порциями по {batch_size} строк')
        for batch in parquet.iter_batches(batch_size=batch_size, columns=cols, use_threads=False):
            # В Python преобразуется только текущая порция объявлений.
            buffer = []
            for row in batch.to_pylist():
                item_id = row['item_id']
                if not isinstance(item_id, str) or not re.fullmatch('[0-9a-f]{16}', item_id):
                    raise ValueError(f'benchmark_items: некорректный item_id')
                if item_id in seen:
                    raise ValueError('Повторяющийся item_id в benchmark_items')
                seen.add(item_id)
                title = '' if row['item_title_raw'] is None else str(row['item_title_raw'])
                ratings = []
                for name in optional:
                    value = row.get(name)
                    ratings.append(0.0 if value is None or pd.isna(value) else float(value))
                buffer.append((item_id, title,
                               clipped_words(row['item_infm_params_text'], cfg['text']['params_words']),
                               clipped_words(row['item_description_raw'], cfg['text']['description_words']),
                               category(row['item_location_id']), category(row['item_category_id']), *ratings))
            db.executemany('INSERT INTO items VALUES (?,?,?,?,?,?,?,?)', buffer)
            previous = processed // 25000
            processed += batch.num_rows
            if processed // 25000 != previous:
                db.commit()
                log(f'Корпус: прочитано {processed:,} строк, уникальных объявлений {len(seen):,}')
        db.commit()
    os.replace(temporary, destination)
    log(f'Корпус на диске готов: {len(seen):,} объявлений')
    return ItemCorpus(destination)
