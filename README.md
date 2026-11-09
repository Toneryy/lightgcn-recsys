# LightGCN на Gowalla и Yelp2018

Проект по дисциплине «Рекомендательные системы» (ИТМО, 2026), подгруппа 6: Мещеряков Д.П., Соболь В.В.

Реализация LightGCN ([He et al., SIGIR 2020](https://arxiv.org/abs/2002.02126)) на PyTorch и проверка на двух открытых наборах в разбиениях из статьи:

- Gowalla: https://huggingface.co/datasets/reczoo/Gowalla_m1
- Yelp2018: https://huggingface.co/datasets/reczoo/Yelp18_m1

Задача – top-20 рекомендации по неявной обратной связи. Метрики – Recall@20 (основная) и NDCG@20 при полном ранжировании по всем объектам, просмотренные в обучении объекты исключаются. Модель берется после последней эпохи, тест для выбора эпохи не используется.

## Результаты

| Набор | Модель | Recall@20 | NDCG@20 |
|---|---|---|---|
| Gowalla | MostPop | 0.0416 | 0.0317 |
| Gowalla | MF-BPR (K = 0) | 0.1439 | 0.1144 |
| Gowalla | LightGCN (K = 3) | 0.1821 | 0.1552 |
| Gowalla | LightGCN (K = 3), статья | 0.1823 | 0.1555 |
| Yelp2018 | MostPop | 0.0125 | 0.0101 |
| Yelp2018 | MF-BPR (K = 0) | 0.0473 | 0.0380 |
| Yelp2018 | LightGCN (K = 3) | 0.0636 | 0.0523 |
| Yelp2018 | LightGCN (K = 3), статья | 0.0639 | 0.0525 |

Кроме точности считаются покрытие каталога, Gini, доля популярных объектов в выдаче и качество по группам пользователей (`results/*.json`, блокнот `LightGCN.ipynb`).

## Файлы

- `data.py` – загрузка данных с Hugging Face, выборка троек для BPR, нормированная матрица смежности
- `model.py` – LightGCN (при `layers=0` это MF-BPR) и функция потерь BPR
- `metrics.py` – полное ранжирование, Recall/NDCG, метрики разнообразия
- `train.py` – обучение одной модели, результат в `results/<набор>_<модель>.json`
- `rerank.py` – штраф за популярность на обученной модели
- `run_all.sh` – все эксперименты подряд
- `LightGCN.ipynb` – таблицы и графики по результатам

## Запуск

```bash
pip install -r requirements.txt
python train.py --dataset gowalla --layers 3 --save-runs
python train.py --dataset yelp2018 --model pop
bash run_all.sh
```

Данные скачиваются при первом запуске в `data/`. Настройки по умолчанию как в статье и официальной реализации: d = 64, Adam с lr = 0.001, L2 = 1e-4, пакет 2048, 1000 эпох. На RTX 3070 Laptop эпоха LightGCN идет около 5 с на Gowalla и 10 с на Yelp2018, все эксперименты `run_all.sh` – около 8 часов.
