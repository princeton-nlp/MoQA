import json
import argparse
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def get_short_dev(devfile):
    with open(devfile) as f:
        data = json.load(f)["data"]

    dev_questions = set()
    for d in data:
        dev_questions.add(d["question"])

    return dev_questions


def split(data, percent, devfile, seed=42):
    examples = {}
    for i, d in enumerate(data):
        t = d["question_type"]
        if not t in examples:
            examples[t] = []
        examples[t].append(d)
    dev_questions = get_short_dev(devfile)
    train_stats = {}
    dev_stats = {}
    train_data = []
    dev_data = []
    for k, v in examples.items():
        if k != "short":
            train, dev = train_test_split(v, train_size=percent, random_state=seed)
        else:
            # use original split for short answers with length < 5 tokens
            train = []
            dev = []
            for ex in v:
                if ex["question"] in dev_questions:
                    dev.append(ex)
                else:
                    train.append(ex)
        train_stats[k] = len(train)
        dev_stats[k] = len(dev)
        train_data += train
        dev_data += dev

    train = {"data": train_data, "stats": train_stats}
    dev = {"data": dev_data, "stats": dev_stats}
    return train, dev


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and development sets and keeping the same ratio among each question type.")
    parser.add_argument("--input_file", type=str, default=None,
            help="The input file containing the original training set to split.")
    parser.add_argument("--orqa_dev_file", type=str, default=None,
            help="The file containing the developement set from the orqa split to use for splitting questions with short answers.")
    parser.add_argument("--output_train_file", type=str, default=None,
            help="The output file for the training set.")
    parser.add_argument("--output_dev_file", type=str, default=None,
            help="The output file for the development set.")
    parser.add_argument("--percent", type=float, default=0.8,
            help="The percentage of the data that gets split into the training set.")
    parser.add_argument("--seed", type=int, default=42,
            help="Random seed for reproducibility.")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input_file}")
    with open(args.input_file) as f:
        data = json.load(f)["data"]

    logger.info(f"Splitting data...")
    train, dev = split(data, args.percent, args.orqa_dev_file, args.seed)
    logger.info(f"Splitting data done.")
    logger.info(f"Train stats {train['stats']}")
    logger.info(f"Dev stats {dev['stats']}")
    logger.info(f"Writing to {args.output_train_file}, this might take a couple minutes...")
    with open(args.output_train_file, "w") as f:
        json.dump(train, f)
    logger.info(f"Writing to {args.output_train_file} done.")
    logger.info(f"Writing to {args.output_dev_file}, this might take a couple minutes...")
    with open(args.output_dev_file, "w") as f:
        json.dump(dev, f)
    logger.info(f"Writing to {args.output_dev_file} done.")

