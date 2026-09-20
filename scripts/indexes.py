"""Дисковые индексы BM25, символьного TF-IDF и истории взаимодействий.

Статистики вычисляются по всему корпусу. Блоки задают способ хранения
и не ограничивают множество объявлений, участвующих в поиске.
"""
from array import array
from collections import Counter, OrderedDict
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix, csc_matrix

from data import log


class DiskHistory:
    """Уникальные пары «полный контекст запроса — item_id» из train."""
    columns = ['q', 'loc', 'delivery', 'filters', 'cat', 'item_id']

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self._frame = None
        self._lookup = None
        self._popularity = None

    def __getstate__(self):
        return {'path': self.path, '_frame': None, '_lookup': None, '_popularity': None}

    def _load(self):
        if self._frame is None:
            self._frame = pq.ParquetFile(self.path).read(use_threads=False).to_pandas(types_mapper=pd.ArrowDtype)
        if self._lookup is None:
            self._lookup = {}
            offset = 0
            for key, size in self._frame.groupby('q', sort=False).size().items():
                self._lookup[key] = (offset, offset + int(size))
                offset += int(size)
        if self._popularity is None:
            self._popularity = self._frame['item_id'].value_counts(sort=False).to_dict()

    def fit(self, train, allowed_ids):
        from retrievers import SEARCH_COLUMNS, query_key
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.building.parquet')
        schema = pa.schema([(column, pa.string()) for column in self.columns])
        buffer = []
        with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
            for number, values in enumerate(train[SEARCH_COLUMNS + ['item_id']].itertuples(index=False, name=None), 1):
                if values[-1] in allowed_ids:
                    key = query_key(dict(zip(SEARCH_COLUMNS, values[:-1])))
                    buffer.append(dict(zip(self.columns, (*key, values[-1]))))
                if len(buffer) >= 2048:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
                if number % 50000 == 0:
                    log(f'История: обработано {number:,} пар')
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        # Повторы одной пары контекст–объявление не увеличивают ее вес в истории.
        frame = pq.ParquetFile(temporary).read(use_threads=False).to_pandas(types_mapper=pd.ArrowDtype)
        self._frame = frame.drop_duplicates().sort_values('q', kind='stable').reset_index(drop=True)
        del frame, buffer
        self._frame.to_parquet(temporary, index=False, compression='zstd')
        temporary.replace(path)
        self._lookup = self._popularity = None
        self._load()
        log(f'Компактная история готова: {len(self._frame):,} уникальных пар')
        return self._popularity

    def rankings(self, key, limit):
        """Выдачи по полному контексту, тексту с локацией и только тексту."""
        self._load()
        bounds = self._lookup.get(key[0])
        if bounds is None:
            return [[], [], []]
        frame = self._frame.iloc[bounds[0]:bounds[1]]
        local = frame.loc[frame['loc'].eq(key[1])]
        exact = local.loc[local['delivery'].eq(key[2]) & local['filters'].eq(key[3]) & local['cat'].eq(key[4])]
        result = []
        for subset in [exact, local, frame]:
            counts = subset['item_id'].value_counts(sort=False)
            ranked = sorted(counts.items(), key=lambda pair: (-pair[1], -self._popularity[pair[0]], pair[0]))
            result.append([item_id for item_id, _ in ranked[:limit]])
        return result


class DiskTextIndex:
    """Разреженный текстовый индекс, разбитый на блоки объявлений."""
    def __init__(self, kind, folder, shard_rows=1024):
        self.kind = kind
        self.folder = str(Path(folder).resolve())
        self.shard_rows = int(shard_rows)
        if self.shard_rows < 1:
            raise ValueError('memory.index_shard_rows должен быть положительным')
        self.vocab = {}
        self.shards = []
        self._cache = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Файлы с весами хранятся отдельно и не копируются в index.pkl.
        state['_cache'] = OrderedDict()
        return state

    def fit(self, documents, analyzer, max_features=None):
        folder = Path(self.folder)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        self._cache.clear()
        df, frequencies = array('I'), array('Q')
        row_indices, term_ids, values = array('I'), array('I'), array('f')
        lengths = []
        total_length = 0
        offset = 0

        def flush():
            nonlocal row_indices, term_ids, values, lengths, offset
            if not lengths:
                return
            prefix = folder / f'{len(self.shards):05d}'
            # В блоке храним только присутствующие в нем признаки.
            terms, local_ids = np.unique(np.asarray(term_ids, dtype=np.uint32), return_inverse=True)
            matrix = coo_matrix((np.asarray(values, dtype=np.float32),
                                 (np.asarray(row_indices, dtype=np.uint32), local_ids)),
                                shape=(len(lengths), len(terms))).tocsc()
            for suffix, data in [('terms', terms), ('indptr', matrix.indptr),
                                 ('indices', matrix.indices), ('data', matrix.data),
                                 ('lengths', np.asarray(lengths, dtype=np.float32))]:
                np.save(str(prefix) + '_' + suffix + '.npy', data, allow_pickle=False)
            self.shards.append((prefix.name, offset, len(lengths)))
            offset += len(lengths)
            row_indices, term_ids, values = array('I'), array('I'), array('f')
            lengths = []

        log(f'{self.kind}: построение блоками по {self.shard_rows} объявлений')
        for number, document in enumerate(documents, start=1):
            counts = Counter(analyzer(document))
            if self.kind == 'bm25' and not counts:
                counts['emptydocumenttoken'] = 1
            length = sum(counts.values())
            total_length += length
            row = len(lengths)
            lengths.append(length)
            for token, count in counts.items():
                token_id = self.vocab.get(token)
                if token_id is None:
                    token_id = len(self.vocab)
                    self.vocab[token] = token_id
                    df.append(0)
                    frequencies.append(0)
                df[token_id] += 1
                frequencies[token_id] += count
                row_indices.append(row)
                term_ids.append(token_id)
                values.append(count)
            if len(lengths) >= self.shard_rows:
                flush()
            if number % 10000 == 0:
                log(f'{self.kind}: обработано {number:,} объявлений, словарь {len(self.vocab):,}')
        flush()
        self.n_docs = offset
        if not offset:
            raise ValueError('Пустой корпус индекса')
        self.average_length = total_length / offset
        dfs = np.asarray(df, dtype=np.float32)
        if self.kind == 'bm25':
            self.idf = np.log(1 + (offset - dfs + 0.5) / (dfs + 0.5)).astype(np.float32)
            selected = np.ones(len(self.vocab), dtype=bool)
        else:
            self.idf = (np.log((offset + 1) / (dfs + 1)) + 1).astype(np.float32)
            selected = np.ones(len(self.vocab), dtype=bool)
            if max_features and len(self.vocab) > max_features:
                # Выбираем самые частые по всему корпусу n-граммы, как в CountVectorizer.
                lexical = np.asarray([self.vocab[t] for t in sorted(self.vocab)], dtype=np.int64)
                frequency_array = np.asarray(frequencies, dtype=np.int64)
                best = lexical[(-frequency_array[lexical]).argsort()[:max_features]]
                selected[:] = False
                selected[best] = True
                self.vocab = {token: i for token, i in self.vocab.items() if selected[i]}
        del df, frequencies, dfs
        log(f'{self.kind}: глобальные статистики готовы; финализация {len(self.shards)} блоков')
        for number, (name, _, n_rows) in enumerate(self.shards, start=1):
            prefix = str(folder / name)
            terms = np.load(prefix + '_terms.npy')
            matrix = csc_matrix((np.load(prefix + '_data.npy'),
                                 np.load(prefix + '_indices.npy'),
                                 np.load(prefix + '_indptr.npy')),
                                shape=(n_rows, len(terms)))
            lengths_array = np.load(prefix + '_lengths.npy')
            if self.kind == 'char':
                keep = selected[terms]
                matrix = matrix[:, keep]
                terms = terms[keep]
            repeated_idf = np.repeat(self.idf[terms], np.diff(matrix.indptr))
            if self.kind == 'bm25':
                # BM25: k1=1.5, b=0.75; вариант формулы соответствует bm25s(method='lucene').
                denominator = matrix.data + 1.5 * (0.25 + 0.75 * lengths_array[matrix.indices] / self.average_length)
                matrix.data = (matrix.data / denominator * repeated_idf).astype(np.float32)
            else:
                matrix.data *= repeated_idf
                norms = np.sqrt(np.bincount(matrix.indices,
                                           weights=matrix.data.astype(np.float64) ** 2,
                                           minlength=n_rows))
                matrix.data /= np.maximum(norms[matrix.indices], 1e-30).astype(np.float32)
            for suffix, data in [('terms', terms), ('indptr', matrix.indptr),
                                 ('indices', matrix.indices), ('data', matrix.data)]:
                np.save(prefix + '_' + suffix + '.npy', data, allow_pickle=False)
            Path(prefix + '_lengths.npy').unlink()
            if number % 25 == 0:
                log(f'{self.kind}: готово {number}/{len(self.shards)} блоков')
        log(f'{self.kind}: индекс готов, {offset:,} объявлений, {len(self.vocab):,} признаков')
        return self

    def _load(self, name):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        prefix = str(Path(self.folder) / name)
        data = tuple(np.load(prefix + '_' + suffix + '.npy', mmap_mode='r')
                     for suffix in ['terms', 'indptr', 'indices', 'data'])
        self._cache[name] = data
        # Одновременно держим открытыми не более четырех блоков индекса.
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return data

    def score_many(self, queries, analyzer):
        """Оценки всех объявлений для небольшой порции запросов."""
        prepared = []
        for query in queries:
            counts = Counter(analyzer(query))
            pairs = [(self.vocab[t], float(count)) for t, count in counts.items() if t in self.vocab]
            ids = np.asarray([p[0] for p in pairs], dtype=np.uint32)
            weights = np.asarray([p[1] for p in pairs], dtype=np.float32)
            if self.kind == 'char' and len(ids):
                weights *= self.idf[ids]
                weights /= max(float(np.linalg.norm(weights)), 1e-30)
            prepared.append((ids, weights))
        scores = np.zeros((len(queries), self.n_docs), dtype=np.float32)
        for name, offset, _ in self.shards:
            terms, indptr, indices, values = self._load(name)
            if not len(terms):
                continue
            for row, (ids, weights) in enumerate(prepared):
                positions = np.searchsorted(terms, ids)
                for term_id, pos, weight in zip(ids, positions, weights):
                    if pos >= len(terms) or terms[pos] != term_id:
                        continue
                    begin, end = indptr[pos:pos + 2]
                    scores[row, indices[begin:end] + offset] += values[begin:end] * weight
        return scores
