import argparse
import json
from functools import partial

import numpy as np
import pandas as pd


def f_flip(x, name):
    first = x["first"]
    second = x["second"]
    score = x["score"]

    if second == name:
        first, second = second, first
        score *= -1.0

    return pd.Series([first, score, second],
                     index=["first", "score", "second"])


def get_experiment_stat(data, rounds=10000):
    def calc_p_value(reference_value, choice_list, size, rounds):
        diff_list = []
        for i in range(rounds):
            s_choice = np.random.choice(choice_list, replace=True, size=size)
            if reference_value != 0.0:
                diff_list.append(reference_value - s_choice.mean())
            else:
                diff_list.append(s_choice.mean())
        percentiles = np.round(np.percentile(a=diff_list, q=[5, 95]), 1)
        a = np.array(diff_list)
        p = a[a > 0].shape[0] / a.shape[0]
        p = min(p, 1 - p)
        p = round(p * 100.0, 1)
        return round(np.mean(a), 1), list(percentiles), p

    win = int(data.loc["win"])
    draw = int(data.loc["draw"])
    lost = int(data.loc["lost"])

    choice_list = draw * [50.0] + lost * [0.0] + win * [100.0]
    m_original = sum(choice_list) / len(choice_list)

    size = len(choice_list)
    effect, perc, p = calc_p_value(
        reference_value=0, choice_list=choice_list, size=size, rounds=rounds
    )

    rand_effect, rand_perc, rand_p = calc_p_value(
        reference_value=m_original,
        choice_list=[0.0, 50.0, 100.0],
        size=size,
        rounds=rounds,
    )
    return {
        "win": win,
        "draw": draw,
        "lost": lost,
        "mean_effect": effect,
        "p_value": p,
        "percentiles": perc,
        "rand_effect": rand_effect,
        "rand_p_value": rand_p,
        "rand_percentiles": rand_perc,
    }


def f_compare(x):
    first = x["first"].split("-")[0]
    second = x["second"].split("-")[0]

    if int(first) > int(second):
        return "vs lower"
    else:
        return "vs higher"


def get_result(df):
    result = None
    for name in sorted(df["first"].unique()):
        f1 = partial(f_flip, name=name)
        df_ = df.apply(f1, axis=1)
        df_["name"] = df_.apply(f_compare, axis=1)
        df_["result"] = df_["score"].map(
            {1.0: "win", 0.0: "draw", -1.0: "lost"})

        r = (
            df_[df_["first"] == name]
            .groupby(["first", "result", "name"])["second"]
            .count()
            .unstack()
            .T
        )
        if result is None:
            result = r
        else:
            result = result.merge(r, left_index=True,
                                  right_index=True, how="outer")

    result = result.round(3).T
    result = result.reset_index()
    return result


def get_stats(df):
    stats = []
    for bench in df["first"].unique():
        for compare in ["vs lower", "vs higher"]:
            s = (
                df[df["first"] == bench][["result", compare]]
                .rename(columns={compare: "value"})
                .set_index("result")
            )
            if int(s.isna().sum() == 0):
                results = get_experiment_stat(s, rounds=10000)
                results["name"] = bench
                results["with"] = compare
                stats.append(results)

    stats = pd.DataFrame(stats)
    return stats


def load_stats(filepath):
    with open(filepath, "r") as f:
        raw = json.load(f)

    data = []
    for _, values in raw.items():
        data.extend(values)

    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--stats", default="output/stats.json", type=str)
    parser.add_argument(
        "-o", "--output", default="output/score.md", type=str)
    args, _ = parser.parse_known_args()

    df_raw = load_stats(args.stats)
    df_group = get_result(df_raw)
    stats = get_stats(df_group)

    (
        stats[
            [
                "name",
                "with",
                "win",
                "draw",
                "lost",
                "mean_effect",
                "percentiles",
                "rand_effect",
                "rand_p_value",
                "rand_percentiles",
            ]
        ]
        .rename(
            columns={
                "name": "sample",
                "with": "compare with",
                "mean_effect": "MeanEffect",
                "p_value": "Pvalue",
                "percentiles": "95% CI",
                "rand_effect": "CleanEffect",
                "rand_p_value": "Pvalue (CE)",
                "rand_percentiles": "95% CI (CE)",
            }
        )
        .to_markdown(args.output, index=False)
    )
    print(f"file with results created ({args.output})")


if "__main__" in __name__:
    main()
