"""Поиск кандидатов по тексту, локации и истории взаимодействий.

Лемматизация и RRF адаптированы из rafsimoms/retrieval-search-pipeline;
атрибуция исходного проекта сохранена в README.
"""
from functools import lru_cache
from pathlib import Path
from itertools import islice, chain
import pickle
import re

import numpy as np
import pandas as pd
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer

from data import log
from indexes import DiskTextIndex, DiskHistory

SEARCH_COLUMNS = [
    'search_query', 'search_location_id', 'search_is_delivery_search',
    'search_infm_params_text', 'search_category',
]
token_re = re.compile(r'[а-яёa-z0-9]+')
_morph = None


def text(value):
    return '' if pd.isna(value) else str(value)


def normalize(value):
    # Нормализация применяется к текстам, никогда к query_id/item_id.
    return ' '.join(token_re.findall(text(value).lower().replace('ё', 'е')))


def category(value):
    if pd.isna(value):
        return '<missing>'
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def query_key(row):
    return (normalize(row['search_query']), category(row['search_location_id']),
            category(row['search_is_delivery_search']),
            normalize(row['search_infm_params_text']), category(row['search_category']))


@lru_cache(maxsize=200_000)
def lemma(token):
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph.parse(token)[0].normal_form


def tokenize(value):
    # Для BM25 слова приводятся к начальной форме; повторные разборы берутся из кеша.
    return [lemma(token) for token in token_re.findall(text(value).lower())]


_char_analyzer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).build_analyzer()


def char_tokens(value):
    return _char_analyzer(normalize(value))


def top_indices(scores, k, positions=None):
    """Top-k без полной сортировки корпуса; одинаковые скоры разрешаются по индексу."""
    positions = np.arange(len(scores)) if positions is None else np.asarray(positions)
    values = scores[positions]
    valid = np.isfinite(values) & (values > 0)
    positions, values = positions[valid], values[valid]
    if not len(positions) or k <= 0:
        return np.empty(0, dtype=np.int64)
    if len(positions) > k:
        boundary = np.partition(values, len(values) - k)[len(values) - k]
        better = positions[values > boundary]
        ties = np.sort(positions[values == boundary])[:k - len(better)]
        positions = np.concatenate([better, ties])
    return positions[np.lexsort((positions, -scores[positions]))]


def reciprocal_rank_fusion(rankings, weights, rrf_k=60):
    """Сложить вклады каналов по позициям объявлений в их выдаче."""
    fused = {}
    for ranked, weight in zip(rankings, weights):
        for rank, item_id in enumerate(dict.fromkeys(ranked), start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """BM25 + символьный поиск + история; независимые глобальные и локальные списки."""
    def __init__(self, config):
        self.config = config

    def fit(self, corpus, train, work_dir):
        """Построить текстовые индексы и историю по доступному корпусу."""
        items = corpus.metadata()
        self.item_ids = items['item_id'].to_numpy()
        self.id_to_pos = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.locations = items['item_location_id'].map(category).to_numpy()
        self.categories = items['item_category_id'].map(category).to_numpy()
        self.local_positions = {key: np.asarray(indices) for key, indices in
                                pd.Series(self.locations).groupby(self.locations).groups.items()}
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        shard_rows = self.config.get('memory', {}).get('index_shard_rows', 1024)
        self.bm25 = DiskTextIndex('bm25', work_dir / 'bm25', shard_rows).fit(
            corpus.documents('bm25', self.config['text']['title_repeat']), tokenize)
        self.char_index = None
        if self.config['search']['use_char']:
            self.char_index = DiskTextIndex('char', work_dir / 'char', shard_rows).fit(
                corpus.documents('char'), char_tokens,
                max_features=self.config['search']['char_max_features'])
        self.history = None
        self.popularity = {}
        if self.config['search']['use_history']:
            self.history = DiskHistory(work_dir / 'history.parquet')
            self.popularity = self.history.fit(train, self.id_to_pos)
        # Резервная выдача заполняет оставшиеся места, если текстовых совпадений мало.
        fallback = items[['item_id']].copy()
        fallback['popularity'] = fallback['item_id'].map(self.popularity).fillna(0)
        for col in ['item_rating_reviews_count', 'item_rating']:
            values = items[col] if col in items else pd.Series(0, index=items.index)
            fallback[col] = pd.to_numeric(values, errors='coerce').fillna(0)
        self.fallback = fallback.sort_values(
            ['popularity', 'item_rating_reviews_count', 'item_rating', 'item_id'],
            ascending=[False, False, False, True]).index.to_numpy()
        return self

    def search(self, row, prepared=None):
        """Объединить каналы и вернуть до 50 уникальных item_id."""
        s = self.config['search']
        key = query_key(row)
        filters = text(row['search_infm_params_text']).strip()
        local = self.local_positions.get(key[1], np.empty(0, dtype=np.int64))
        if key[1] == '<missing>':
            local = np.empty(0, dtype=np.int64)
        rankings, weights = [], []

        def add_scores(name, scores, k):
            rankings.append(self.item_ids[top_indices(scores, k)].tolist())
            weights.append(s['weights'][name])
            if s['use_local'] and len(local):
                # Top-k ВНУТРИ локации, а не фильтрация глобального top-k.
                rankings.append(self.item_ids[top_indices(scores, s['local_k'], local)].tolist())
                weights.append(s['weights'][name + '_local'])

        prepared = self._prepare([row])[0] if prepared is None else prepared
        add_scores('bm25', prepared['bm25'], s['global_k'])
        if self.char_index is not None:
            add_scores('char', prepared['char'], s['char_k'])
        if filters and text(row['search_query']).strip():
            # Общие слова фильтров не должны вытеснить слова самого запроса.
            scores = prepared['filters']
            rankings.append(self.item_ids[top_indices(scores, s['global_k'])].tolist())
            weights.append(s['weights']['query_with_filters'])
        if self.history is not None and key[0]:
            for ranked, weight_name in zip(self.history.rankings(key, s['history_k']),
                                          ['history_context', 'history_text_location', 'history_text']):
                rankings.append(ranked)
                weights.append(s['weights'][weight_name])
        fused = reciprocal_rank_fusion(rankings, weights, s['rrf_k'])
        adjusted = []
        for item_id, score in fused:
            item_category = self.categories[self.id_to_pos[item_id]]
            if key[4] not in ('0', '<missing>') and item_category != key[4]:
                score *= s['category_mismatch_multiplier']
            adjusted.append((item_id, score))
        pool = [item_id for item_id, _ in sorted(adjusted, key=lambda pair: (-pair[1], pair[0]))]
        predictions = pool[:50]
        seen = set(predictions)
        if len(predictions) < min(50, len(self.item_ids)):
            local_set = set(local) if s['use_local'] else set()
            order = chain((i for i in self.fallback if i in local_set), self.fallback)
            for pos in order:
                item_id = self.item_ids[pos]
                if item_id not in seen:
                    predictions.append(item_id)
                    seen.add(item_id)
                if len(predictions) == min(50, len(self.item_ids)):
                    break
        return predictions

    def _prepare(self, rows):
        """Рассчитать текстовые оценки для небольшой порции запросов."""
        # Если текст пуст, используем фильтры как поисковую строку.
        queries = [text(r['search_query']).strip() or text(r['search_infm_params_text']).strip() for r in rows]
        bm25 = self.bm25.score_many(queries, tokenize)
        result = [{'bm25': scores} for scores in bm25]
        if self.char_index is not None:
            for record, scores in zip(result, self.char_index.score_many(queries, char_tokens)):
                record['char'] = scores
        positions = [i for i, r in enumerate(rows) if text(r['search_query']).strip() and text(r['search_infm_params_text']).strip()]
        if positions:
            expanded = [queries[i] + ' ' + text(rows[i]['search_infm_params_text']).strip() for i in positions]
            for i, scores in zip(positions, self.bm25.score_many(expanded, tokenize)):
                result[i]['filters'] = scores
        return result

    def predict(self, queries):
        """Обработать запросы в исходном порядке, ограничивая размер порции."""
        predictions = []
        batch_size = int(self.config.get('memory', {}).get('query_batch_rows', 8))
        if batch_size < 1:
            raise ValueError('memory.query_batch_rows должен быть положительным')
        iterator = queries.itertuples(index=False, name=None)
        columns = list(queries.columns)
        while True:
            rows = [dict(zip(columns, values)) for values in islice(iterator, batch_size)]
            if not rows:
                break
            prepared = self._prepare(rows)
            for row, scores in zip(rows, prepared):
                predictions.append(self.search(row, scores))
            # Освобождаем оценки текущей порции перед обработкой следующей.
            del prepared, scores
            log(f'Поиск: {len(predictions)}/{len(queries)} запросов')
        return predictions

    def save(self, path):
        """Сохранить параметры модели и ссылки на дисковые индексы."""
        with open(path, 'wb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        # Загружать только собственные локально построенные индексы.
        with open(path, 'rb') as stream:
            model = pickle.load(stream)
        root = Path(path).resolve().parent / 'index_data'
        if root.exists():
            model.bm25.folder = str(root / 'bm25')
            if model.char_index is not None:
                model.char_index.folder = str(root / 'char')
            if model.history is not None:
                model.history.path = str(root / 'history.parquet')
        return model
