from collections import defaultdict
import json
import os
import logging
import time
import argparse
import csv

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter long questions based on the mechnical turk annotations.")
    parser.add_argument("--data_file", type=str, default=None,
            help="The data file.")
    parser.add_argument("--annotation_file", type=str, default=None,
            help="Mechanical turk annotation file.")
    parser.add_argument("--num_short", type=int, default=0,
            help="The maximum number of allowed short annotations.")
    args = parser.parse_args()

    with open(args.annotation_file) as f:
        annotations = json.load(f)

    for a in annotations:
        a["num_short"] = len([c for c in a["anno"] if c["choice"] == "short"])
    good_ids = set([a["id"] for a in annotations if a["num_short"] <= args.num_short])

    with open(args.data_file) as f:
        output = json.load(f)
        data = output.pop("data")

    stats = defaultdict(lambda: 0)
    for d in data:
        stats[d["question_type"]] += 1
    print(f"before filter stats: {dict(stats)}")

    new_data = [d for d in data if d["id"] in good_ids or d["question_type"] != "long"]
    output["data"] = new_data

    stats = defaultdict(lambda: 0)
    for d in new_data:
        stats[d["question_type"]] += 1
    print(f"after filter stats: {dict(stats)}")

    with open(f"{args.data_file.replace('.json', '')}-filtered.json", "w") as f:
        json.dump(output, f)

