import argparse
from collections import defaultdict
import concurrent.futures
import gzip
import json
import logging
import os
import re
import time
import unicodedata

import bs4
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# qa hard em https://github.com/shmsw25/qa-hard-em/blob/master/split_nq.py#L28
def qa_hard_em_process_line(line):
    question = line["question_text"]
    document = [t['token'] for t in line['document_tokens']]
    answers = []
    for annotations in line["annotations"]:
        for short_annotation in annotations["short_answers"]:
            if short_annotation["end_token"] - short_annotation["start_token"] > 5:
                continue
            answer = document[short_annotation["start_token"]:short_annotation["end_token"]]
            answers.append(" ".join(answer))

    return line["example_id"], question, list(set(answers)), line["document_title"]


# orqa https://github.com/google-research/language/blob/master/language/orqa/preprocessing/convert_to_nq_open.py
def orqa_process_line(line):
    question_text = line["question_text"]

    # Convert to bytes so that we can index by byte offsets from the data.
    document_html = line["document_html"].encode("utf-8")

    answer_texts = set()
    for annotation in line["annotations"]:
        for sa in annotation["short_answers"]:
            if sa["end_token"] - sa["start_token"] <= 5:
                raw_html = document_html[sa["start_byte"]:sa["end_byte"]]
                answer_texts.add(bs4.BeautifulSoup(raw_html, "lxml").get_text())

    return line["example_id"], question_text, list(answer_texts), line["document_title"]

def include_example(example) -> bool:
    """
        Decide if the example should be included in the dataset.
        Based on if the title include a set of strings
    """
    if re.match(f"(List of .+)|(Index of .+)|(Outline of.+)", example["document_title"]):
        return False
    if "(disambiguation)" in example["document_title"].lower() or "(disambiguation page)" in example["document_title"].lower():
        return False
    return True

def has_structure(html):
    """
        Check if an html box is just text or has other structure
        Specifically, we count the table (and related) tags as structures
        but lists are fine (they tend to preserve natural-paragraph-like text even when concatenated together).
    """
    return html.table is not None or html.td is not None or html.tr is not None or html.ol is not None
    #return html.table is not None or html.td is not None or html.ul is not None or html.dl is not None or html.tr is not None or html.ol is not None

def include_long(answer, answer_html, min_tokens=10, max_tokens=500) -> bool:
    """
        Decide if the long answer should be included.
        Based on if the long answer is long enough and if it's part of some html tags.
    """
    num_tokens = answer["end_token"] - answer["start_token"]
    if num_tokens < min_tokens or num_tokens > max_tokens:
        return False
    if has_structure(answer_html):
        return False
    return True

def process_line(line, stats=None, null_answers_threshold=0.4, max_long_answer_tokens=500):
    """
        Process the example to get all relevant information. Can also include stats about the question and answers.
    """
    include = include_example(line)

    question_text = line["question_text"]

    # Convert to bytes so that we can index by byte offsets from the data.
    document_html = line["document_html"].encode("utf-8")

    num_short_answers = 0
    num_medium_answers = 0
    num_long_answers = 0
    num_yn_answers = 0

    yes_no_answer = []
    answer_texts = set()
    long_answer_texts = []
    long_answer_valid = []
    long_answer_natural = []
    num_annotations = len(line["annotations"])

    for annotation in line["annotations"]:
        # Check if a long answer actually exists
        if annotation["long_answer"]["candidate_index"] != -1:
            # extract answers using bs4
            long_answer_html = document_html[
                annotation["long_answer"]["start_byte"]:annotation["long_answer"]["end_byte"]
            ]
            long_answer_html = bs4.BeautifulSoup(long_answer_html, "lxml")

            long_answer_natural.append(not has_structure(long_answer_html))
            long_answer_valid.append(include_long(annotation["long_answer"], long_answer_html, max_tokens=max_long_answer_tokens))

            # remove the support tags for references (e.g. "info [123]")
            while long_answer_html.find("sup"):
                long_answer_html.find("sup").decompose()
            long_answer = long_answer_html.get_text().strip()
            #long_answer = unicodedata.normalize("NFKD", long_answer).strip()

            if not long_answer_valid[-1]:
                # normalize for whitespace if the long answer wasn't valid
                # -> won't be affected in standardizing type 4 questions
                long_answer = " ".join(long_answer.split()[:max_long_answer_tokens])
            long_answer_texts.append(long_answer)

            # we found a long answer but may or may not have been included
            num_long_answers += 1

            if annotation["yes_no_answer"] != "NONE":
                num_yn_answers += 1
                yes_no_answer.append(annotation["yes_no_answer"])

            for sa in annotation["short_answers"]:
                raw_html = document_html[sa["start_byte"]:sa["end_byte"]]
                answer_html = bs4.BeautifulSoup(raw_html, "lxml")
                while answer_html.find("sup"):
                    answer_html.find("sup").decompose()
                unicode_str = answer_html.get_text().strip()
                #normalized = unicodedata.normalize("NFKD", unicode_str).strip()
                if len(unicode_str) == 0:
                    continue
                answer_texts.add(unicode_str)

                if sa["end_token"] - sa["start_token"] <= 5:
                    num_short_answers += 1

                else:
                    num_medium_answers += 1

    if "YES" in yes_no_answer and "NO" in yes_no_answer:
        #logger.info(f"Example {line['example_id']} has a contradicting yes no answer, excluding from dataset.")
        include = False

    if num_short_answers > 0:
        q_type = "short"
        # keep the subset of open-domain nq from ORQA
        include = True
    elif num_long_answers / num_annotations < null_answers_threshold:
        q_type = "null"
    elif num_yn_answers > 0:
        q_type = "yesno"
    elif num_medium_answers > 0:
        q_type = "medium"
    elif num_long_answers > 0:
        # for type 4 questions, we only restrict the long answers to longer than 10 tokens and not from tables
        long_answer_texts = [a for (a, v) in zip(long_answer_texts, long_answer_valid) if v]
        long_answer_natural = [a for (a, v) in zip(long_answer_natural, long_answer_valid) if v]

        if len(long_answer_texts) == 0:
            # there is a case where there are long answers that were excluded but there were also no short/yn answers,
            # so instead of counting them as NULL answers(Type 5), we exclude this question
            include = False
        q_type = "long"
    else:
        q_type = "null"

    if len(long_answer_texts) != len(long_answer_natural):
        raise Exception("length mismatch")

    output = {
        "id": str(line["example_id"]),
        "question": question_text,
        "answers": list(answer_texts),
        "long_answers": list(long_answer_texts),
        "long_answers_from_natural_paragraph": list(long_answer_natural),
        "yes_no_answer": yes_no_answer,
        "titles": [line["document_title"]],
        "question_type": str(q_type),
    }

    return output if include else None


def process_file(input_file):
    logger.info(f"Processing file {input_file}.")
    examples = []
    with gzip.open(input_file) as f:
        for line in tqdm(f, desc=f"Processing {input_file}"):
            json_example = json.loads(line)
            output = process_line(json_example)
            if output is not None:
                examples.append(output)
    logger.info(f"Finished preprocessing {input_file}.")
    return examples


def process_dir(input_dir, num_process=1):
    examples = []
    files = [os.path.join(input_dir, i) for i in sorted(os.listdir(input_dir)) if i.endswith("jsonl.gz")]

    if num_process != 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as executor:
            for ex in executor.map(process_file, files):
                examples += ex
            return examples
    else:
        for file in files:
            examples = process_file(file)
        return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process Natural Questions dataset from the original format (NOT the simplified version)")
    parser.add_argument("--input_path", type=str, default=None,
            help="The input path where the dataset is stored (either a directory containing jsonl.gz file(s) or a jsonl.gz file)")
    parser.add_argument("--output_path", type=str, default=None,
            help="The output path where the preprocessed dataset will be stored.")
    parser.add_argument("--include_answer_stats", action="store_true",
            help="Include stats about the answers in the output json with the questions.")
    parser.add_argument("--num_process", type=int, default=1,
            help="The number of processes to use.")
    args = parser.parse_args()

    logger.info(f"Preprocessing {args.input_path}")
    start_time = time.time()
    if args.input_path.endswith("jsonl.gz"):
        all_examples = process_file(args.input_path)
    else:
        all_examples = process_dir(args.input_path, num_process=args.num_process)
    end_time = time.time()

    output = {"data": all_examples}
    stats = defaultdict(lambda: 0)
    for d in all_examples:
        stats[d["question_type"]] += 1
    stats = dict(stats)
    if args.include_answer_stats:
        output["stats"] = stats
    logger.info(
        f"Finished processing {args.input_path} in {end_time - start_time:.2f} seconds, saving to {args.output_path}, final stats: \n"
    )
    for k, v in stats.items():
        print(k, v)

    with open(args.output_path, "w") as f:
        json.dump(output, f)
