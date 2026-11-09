#!/usr/bin/env bash
# все эксперименты отчета подряд (RTX 3070 Laptop: около 8 часов)
set -e
PY=${PY:-python}
for d in gowalla yelp2018; do
  $PY train.py --dataset $d --model pop
  $PY train.py --dataset $d --layers 3 --save-runs
done
for d in gowalla yelp2018; do
  $PY train.py --dataset $d --layers 0
done
# сравнение числа слоев при одинаковом бюджете 200 эпох
for d in gowalla yelp2018; do
  for k in 1 2 4; do
    $PY train.py --dataset $d --layers $k --epochs 200
  done
done
for d in gowalla yelp2018; do
  $PY rerank.py --dataset $d
done
