# Project Monster

Исходное видео [ТАКОГО АПГРЕЙДА ВЕРТУШКИ ТЫ ЕЩЕ НЕ ВИДЕЛ](https://www.youtube.com/watch?v=Qaaw8O41Rak) в котором инкрементально делается улучшение винилового проигрывателя и снимается звук для оценки качества

В каждом эксперименте автор видео, делает разумное улучшение аппаратуры и снимает звук с фонокорректора.

В данном репозитории, сделана попытка получения статистически значимой оценки качества улучшения винилового проигрывателя с помощью слепого прослушивания записей и формирования рейтинга.

Каждый может попробовать самостоятельно и понять на сколько он может определить более качественный звук. Хочу отметить, что если вы хотите получить стат значимые эффекты нужно слушать в хороших наушниках и в тихом помещении.

В нашем анализе, мы исходим из того, что качество звука после апгрейда точно не должно стать ХУЖЕ (т.е. может остаться на том же уровне, а может стать немного лучше)

Каждое улучшение делается автором инкрементально и на все предыдущие изменения

| Файл с записью              | Апгрейд                                                                          |
| --------------------------- | -------------------------------------------------------------------------------- |
| 01-base.wav                 | Базовая запись (Проигрыватель Pro-Ject Debut III DC)                             |
| 02-2MBlue.wav               | Головка звукоснимателя Ortofon 2M Blue Bulk                                      |
| 03-Fezz-Gaia.wav            | Фонокорректор Fezz Audio Gaia Evo, Кабели Audioquest Evergreen, Chord Cobra VEE3 |
| 04-PowerSupply.wav          | Блок питания NoName                                                              |
| 05-subdisc-acrylplatter.wav | Суб-диск Pro-Ject Debut Alu Sub-Platter, Опорный диск Pro-Ject Acryl It          |
| 06-isoacoustics-feet.wav    | Ножки IsoAcoustics Orea Graphite                                                 |
| 07-leather-mat.wav          | Слипмат Tonar Leather Player Mat                                                 |
| 08-cork-mat.wav             | Слипмат No Name пробковый                                                        | 
| 09-clamp.wav                | Клэмп Analog Renaissance Mjöllnir Clamp                                          |


## Установка

1. скачайте архив с треками - https://disk.yandex.ru/d/IGdrt2Q9BsC0QQ и разархивируйте в папку `data/`
2. установите `python` и среду `pip install -r requirements.txt`

## Как работает оценка

Вы запускаете `python src/run_experiments.py` и начинаете слушать семплы музыки по парам. Семплы подаются в случайном порядке. После прошлушивания двух семплов Вам нужно оценить какой семпл лучше (первый, оба хороши или второй). По окончанию экспериментов, на вашем компьютере появится файл с результатами `output/stats.json`.  
 
После проведения экспериментов (минимум 10-и), вы можете перейти к след шагу и оценить статистически значимую оценку для каждого семпла. Для этого запустите `python src/get_score.py` и у Вас создастся файл `output/score.md`.

Оценка сводится в таблицу (файл `output/score.md`, пример таблицы ниже).

Каждый семпл сравнивается в 2х вариантах с семплами более простой сборки (`vs lower`) и более хорошей сборки (`vs higher`)

Пример таблицы ниже:

| sample           | compare with | win | draw | lost | MeanEffect | 95% CI       | CleanEffect | Pvalue (CE) | 95% CI (CE)   |
|:---------------- |:------------ | ---:| ----:| ----:| ----------:|:------------ | -----------:| -----------:|:------------- |
| 01-base.wav      | vs higher    |   9 |    3 |    7 |       55.2 | [36.8, 73.7] |         5.3 |        34.1 | [-10.5, 21.1] |
| 02-2MBlue.wav    | vs lower     |   1 |    2 |    5 |       24.9 | [6.2, 43.8]  |         -25 |         2.5 | [-50.0, 0.0]  |
| 02-2MBlue.wav    | vs higher    |   2 |    1 |    3 |       41.6 | [16.7, 75.0] |        -8.4 |        22.8 | [-33.3, 16.7] |
| 03-Fezz-Gaia.wav | vs lower     |   9 |    2 |    6 |       58.9 | [41.2, 76.5] |         8.8 |        23.6 | [-8.8, 26.5]  |

В таблицу добавлены след результаты:
- win, draw, lost - кол-во побед, ничей и проигрышей данного семпла со всеми другими сборками
- MeanEffect - показывает долю побед данного семпла над другим сборками, не стоит опираться на данную оценку (где 0 - означает всегда проигрывает, 50 - ничья, 100 - всегда побеждает другие сборки)
- 95% CI - показывает 95% доверительный интервал для оценки MeanEffect (это нужно когда сделано мало экспериментов и оценка может иметь большую неопределенность)
- CleanEffect - показывает статистически значимый эффект (данный показатель рассчитывается след образом CleanEffect = MeanEffect - RandomEffect, т.е. из среднего эффекта вычитается случайный эффект (случайный эффект равен эффекту, если мы будет ставить оценки случайным образом)
- P value (CE) - показывает на сколько полученное значения CleanEffect статистически значимо (должно быть меньше 5%, чем меньше тем лучше)
- 95% CI (CE) - показывает 95% доверительный интервал для оценки CleanEffect

Хорошей оценки!
