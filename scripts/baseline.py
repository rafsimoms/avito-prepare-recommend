"""Построение индексов, поиск кандидатов и сохранение answer.csv.

Команды: run — полный запуск, predict — поиск по готовым индексам,
check — проверка формата сохраненного ответа.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from data import build_corpus, log, read_small_table
from retrievers import HybridRetriever, SEARCH_COLUMNS


def load_config(path):
    """Разрешить относительные пути от папки, в которой лежит конфиг."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['paths'] = {key: str((path.parent / value).resolve())
                    for key, value in cfg['paths'].items()}
    return cfg


def check_ids(series, name, unique=True):
    """Проверить исходные строковые ID, сохраняя регистр и ведущие нули."""
    valid = series.map(lambda value: isinstance(value, str) and (
        len(value) == 16 if name == 'query_id'
        else bool(re.fullmatch(r'[0-9a-f]{16}', value))))
    if not valid.all() or (unique and series.duplicated().any()):
        raise ValueError(f'Некорректные или повторяющиеся {name}; нужны исходные строки из Parquet')


def read_data(cfg, command):
    """Прочитать запросы, ID корпуса и необходимые для истории поля train."""
    required = {
        'train': SEARCH_COLUMNS + ['item_id'],
        'items': ['item_id'],
        'queries': ['query_id'] + SEARCH_COLUMNS,
    }
    tables = {'train': None}
    for name, columns in required.items():
        if name == 'train' and command in ('check', 'predict'):
            continue
        path = Path(cfg['paths'][name])
        if not path.exists():
            raise FileNotFoundError(f'Не найден {path}. Укажите путь к Parquet в config.yaml')
        frame = read_small_table(path, columns)
        id_column = 'query_id' if name == 'queries' else 'item_id'
        check_ids(frame[id_column], id_column, unique=name != 'train')
        tables[name] = frame
        log(f'{name}: {len(frame):,} строк')
    return tables['train'], tables['items'], tables['queries']


def validate_answer(answer, queries, items):
    """Проверить все требования задания к query_id и спискам кандидатов."""
    if list(answer.columns) != ['query_id', 'answer']:
        raise ValueError('CSV должен содержать ровно колонки query_id,answer')
    check_ids(answer['query_id'], 'query_id')
    if len(answer) != len(queries) or set(answer['query_id']) != set(queries['query_id']):
        raise ValueError('Набор query_id не совпадает с benchmark_queries')
    allowed = set(items['item_id'])
    for row in answer.itertuples(index=False):
        if not isinstance(row.answer, str):
            raise ValueError(f'{row.query_id}: answer должен быть строкой')
        ids = row.answer.split()
        if len(ids) > 50 or len(ids) != len(set(ids)):
            raise ValueError(f'{row.query_id}: больше 50 ID или есть повторы')
        if any(not re.fullmatch(r'[0-9a-f]{16}', item_id) or item_id not in allowed for item_id in ids):
            raise ValueError(f'{row.query_id}: неизвестный или некорректный item_id')


def save_answer(predictions, queries, items, path):
    """Сохранить UTF-8 CSV без индекса и проверить записанный файл."""
    answer = pd.DataFrame({
        'query_id': queries['query_id'].tolist(),
        'answer': [' '.join(ids) for ids in predictions],
    })
    validate_answer(answer, queries, items)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(path, index=False, encoding='utf-8')
    validate_answer(pd.read_csv(path, dtype=str, keep_default_na=False), queries, items)


def file_hash(path):
    """Вычислить SHA-256 файла по частям, не загружая его целиком."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_signature(cfg):
    """Связать сохраненный индекс с данными, кодом и параметрами запуска."""
    return {
        'data': {name: file_hash(cfg['paths'][name]) for name in ['train', 'items']},
        'code': {name: file_hash(Path(__file__).parent / name)
                 for name in ['baseline.py', 'retrievers.py', 'data.py', 'indexes.py']},
        'text': cfg['text'],
        'search': cfg['search'],
        'memory': cfg.get('memory', {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'predict', 'check'])
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'config.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, items, queries = read_data(cfg, args.command)

    if args.command == 'check':
        answer = pd.read_csv(cfg['paths']['answer'], dtype=str, keep_default_na=False)
        validate_answer(answer, queries, items)
        print(f'CSV корректен: {len(queries)} запросов')
        return

    folder = Path(cfg['paths']['artifacts'])
    signature = cache_signature(cfg)
    if args.command == 'run':
        log(f'Построение индексов: {len(items):,} объявлений, {len(train):,} пар train')
        folder.mkdir(parents=True, exist_ok=True)
        # Индекс считается готовым только после завершения всех этапов fit.
        (folder / 'manifest.json').unlink(missing_ok=True)
        corpus = build_corpus(cfg, folder / 'corpus.sqlite')
        model = HybridRetriever(cfg).fit(corpus, train, folder / 'index_data')
        model.save(folder / 'index.pkl')
        (folder / 'manifest.json').write_text(
            json.dumps(signature, indent=2, ensure_ascii=False), encoding='utf-8')
    else:
        if not (folder / 'manifest.json').exists() or not (folder / 'index.pkl').exists():
            raise ValueError('Индекс не найден: сначала выполните команду run')
        saved = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if saved != signature:
            raise ValueError('Данные, код или параметры изменились: перестройте индекс командой run')
        model = HybridRetriever.load(folder / 'index.pkl')

    predictions = model.predict(queries)
    save_answer(predictions, queries, items, cfg['paths']['answer'])
    log(f'Готово: {cfg["paths"]["answer"]}; {len(queries)} запросов')


if __name__ == '__main__':
    main()
