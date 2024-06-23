import argparse
import datetime
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List, NoReturn, Tuple

import pygame


def update_stats(stats: str, first: str, second) -> Dict[str, str]:
    string = """Select which track are better first or second?
    1. First
    2. Second
    3. Both
    4. Skip this test
Answer:"""
    while True:
        answer = input(string)
        if answer.isdigit() and int(answer) in range(1, 5):
            answer = int(answer)
            break

    if answer == 1:
        score = 1.0
    elif answer == 2:
        score = -1.0
    elif answer == 3:
        score = 0.0
    elif answer == 4:
        return stats
    else:
        raise NotImplementedError()

    last_key = sorted(stats.keys())[-1]
    stats[last_key].append({"first": first, "second": second, "score": score})
    return stats


def play_track(filepath: str) -> NoReturn:
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass


def load_stats(filepath: str) -> Dict[str, str]:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    datenow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    stats[datenow] = []
    return stats


def save_stats(filepath: str, stats: List[Any]) -> NoReturn:
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4, sort_keys=True)


def get_candidates(object_list: List[str],
                   num_tests: int) -> List[Tuple[str, str]]:
    permutations = list(itertools.permutations(object_list, 2))
    candidates_list = [random.choice(permutations) for i in range(num_tests)]
    return candidates_list


def is_bool_input(message: str) -> bool:
    while True:
        answer = input(f"{message} [y/n] ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False


def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_tests", default=10, type=int)
    parser.add_argument("-p", "--path", default="data/", type=str)
    parser.add_argument("-o", "--output",
                        default="output/stats.json", type=str)
    args, _ = parser.parse_known_args()

    # define full list of tracks to testing
    object_list = [
        "01-base.wav",
        "02-2MBlue.wav",
        "03-Fezz-Gaia.wav",
    ]
    # load output file
    results = load_stats(args.output)
    # create list of tracks to compare
    candidates_list = get_candidates(object_list, args.num_tests)

    # run tests, one by one
    for i, (first, second) in enumerate(candidates_list):
        is_continue = is_bool_input("run next test?")
        if not is_continue:
            break

        print(f"Test #{i+1}/{args.num_tests}")

        print("first track playing")
        play_track(os.path.join(args.path, first))

        time.sleep(1.0)
        print("second track playing")
        play_track(os.path.join(args.path, second))

        # get and add score for this test
        update_stats(results, first, second)

        # save current test
        save_stats(args.output, results)


if "__main__" == __name__:
    main()
