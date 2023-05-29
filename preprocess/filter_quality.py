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
    parser = argparse.ArgumentParser(description="Filter questions based on the question quality annotation file.")
    parser.add_argument("--data_file", type=str, default=None,
            help="The data file.")
    parser.add_argument("--quality_file", type=str, default=None,
            help="The quesiton quality annotation file.")
    parser.add_argument("--quality_threshold", type=int, default=None,
            help="The minimum number of good question annotations required to keep the example.")
    args = parser.parse_args()

    question_quality = {}
    with open(args.quality_file) as f:
        csv_reader = csv.reader(f, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f"the columns are {', '.join(row)}")
            else:
                question_quality[row[0].replace("\\xEF\\xBB\\xBF", "\ufeff")] = sum([1 if r == "GOOD_QUESTION" else 0 for r in row[1:]])
            line_count += 1

    with open(args.data_file) as f:
        output = json.load(f)
        data = output.pop("data")

    stats = defaultdict(lambda: 0)
    for d in data:
        stats[d["question_type"]] += 1
    print(f"before filter stats: {dict(stats)}")

    new_data = [d for d in data if question_quality[d["question"]] >= args.quality_threshold or d["question_type"] == "short"]
    output["data"] = new_data

    stats = defaultdict(lambda: 0)
    for d in new_data:
        stats[d["question_type"]] += 1
    print(f"after filter stats: {dict(stats)}")

    with open(f"{args.data_file.replace('.json', '')}-filtered.json", "w") as f:
        json.dump(output, f)

