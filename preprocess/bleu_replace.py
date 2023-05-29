import argparse
import json
import time
import concurrent.futures
import string
import re
import os
from functools import partial
import math

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import NLTKWordTokenizer, word_tokenize

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# from KILT https://github.com/facebookresearch/KILT/blob/main/kilt/kilt_utils.py#L116
def get_bleu(candidate_tokens, gold_tokens):
    candidate_tokens = [x for x in candidate_tokens if len(x.strip()) > 0]
    gold_tokens = [x for x in gold_tokens if len(x.strip()) > 0]

    # The default BLEU calculates a score for up to
    # 4-grams using uniform weights (this is called BLEU-4)
    weights = (0.25, 0.25, 0.25, 0.25)

    if len(gold_tokens) < 4:
        # lower order ngrams
        weights = [1.0 / len(gold_tokens) for _ in range(len(gold_tokens))]

    BLEUscore = sentence_bleu(
        [candidate_tokens], gold_tokens, weights=weights
    )
    return BLEUscore

# modeled after KILT, but simpler https://github.com/facebookresearch/KILT/blob/main/kilt/kilt_utils.py#L196
# we just need to match the answer to some paragraph in the page, return that paragraph and the score
def match_answer(tokenizer, answer, page, bleu_threshold = 0.5, max_paragraph_candidate = 3):
    if len(page) == 0:
        return None, None, None, 0

    original_answer = answer
    answer = normalize_answer(answer)
    answer_tokens = tokenizer.tokenize(original_answer)
    answer_tokens = [normalize_answer(x) for x in tokenizer.tokenize(original_answer)]

    tokenized_paragraphs = []
    tokenized_paragraph_spans = []

    candidate_dict = {}
    debug = None

    for idx, paragraph in enumerate(page):
        index = paragraph.find(original_answer)
        if index >= 0:
            return paragraph, index, index + len(original_answer), 1.0

        index = paragraph.find(answer)
        if index >= 0:
            return paragraph, index, index + len(original_answer), 1.0
        elif normalize_answer(paragraph).find(answer) >= 0:
            debug = paragraph

        paragraph_token_spans = list(tokenizer.span_tokenize(paragraph))
        paragraph_tokens = [paragraph[start:end] for start, end in paragraph_token_spans]
        paragraph_tokens = [normalize_answer(x) for x in paragraph_tokens]

        tokenized_paragraph_spans.append(paragraph_token_spans)
        tokenized_paragraphs.append(paragraph_tokens)

        intersection = len(set(paragraph_tokens).intersection(set(answer_tokens)))

        if intersection == len(answer_tokens):
            # I found all the tokens, let me see if there is a perfect match
            ax = " ".join([x.strip() for x in answer_tokens if len(x.strip()) > 0])
            for w_start in range(len(paragraph_tokens)):
                token = paragraph_tokens[w_start]
                if token == answer_tokens[0]:
                    bx = " ".join(
                        [
                            x.strip()
                            for x in paragraph_tokens[w_start:]
                            if len(x.strip()) > 0
                        ]
                    )
                    if bx.startswith(ax):
                        for w_end in range(w_start, len(paragraph_tokens)):
                            token = paragraph_tokens[w_end]
                            if token == answer_tokens[-1]:
                                cx = " ".join(
                                    [
                                        x.strip()
                                        for x in paragraph_tokens[w_start : w_end + 1]
                                        if len(x.strip()) > 0
                                    ]
                                )
                                if ax == cx:
                                    start_character = paragraph_token_spans[w_start][0]
                                    end_character = paragraph_token_spans[w_end][1]
                                    return paragraph, start_character, end_character, 1.0

        if intersection not in candidate_dict:
            candidate_dict[intersection] = []
        candidate_dict[intersection].append(idx)

    candidate_idx = []
    for key in sorted(candidate_dict.keys(), reverse=True):
        for idx in candidate_dict[key]:
            candidate_idx.append(idx)
        if len(candidate_idx) >= max_paragraph_candidate:
            break

    assert len(candidate_idx) > 0

    max_bleu = 0
    paragraph_idx = None
    start_token = None
    end_token = None

    for idx in candidate_idx:
        paragraph_token_spans = tokenized_paragraph_spans[idx]
        paragraph_tokens = tokenized_paragraphs[idx]

        # perfect match
        for i in range(len(paragraph_tokens) - len(answer_tokens) + 1):
            if paragraph_tokens[i : i + len(answer_tokens)] == answer_tokens:
                start_character = paragraph_token_spans[i][0]
                end_character = paragraph_token_spans[i+len(answer_tokens)-1][1]
                return page[idx], start_character, end_character, 1.0

        # this speeds up our code significantly without having to consider unlikely spans
        span_ratio = 2.0
        min_span_length = int(len(answer_tokens) / span_ratio)
        max_span_length = int(len(answer_tokens) * span_ratio)

        for start in range(len(paragraph_tokens)):
            for end in range(start, len(paragraph_tokens)):
                if end - start < min_span_length or end - start > max_span_length:
                    continue
                candidate = paragraph_tokens[start:end+1]
                bleu = get_bleu(candidate, answer_tokens)
                if bleu >= bleu_threshold and (bleu > max_bleu or (
                    bleu == max_bleu and (start_token is None or (end - start) < end_token - start_token)
                )):
                    max_bleu = bleu
                    paragraph_idx = idx
                    start_token = start
                    end_token = end

        if max_bleu == 1.0:
            break

    if paragraph_idx is not None:
        paragraph = page[paragraph_idx]
        start_character = tokenized_paragraph_spans[paragraph_idx][start_token][0]
        end_character = tokenized_paragraph_spans[paragraph_idx][end_token][1]
    else:
        paragraph = None
        start_character = None
        end_character = None

    return paragraph, start_character, end_character, max_bleu

def replace_examples(title2file, wiki_dir, data):
    new_data = []
    discarded = 0
    title2text = {}
    tokenizer = NLTKWordTokenizer()

    for d in tqdm(data, leave=False):
        title = d["titles"][0]
        qtype = d["question_type"]
        id = d["id"]

        if title in title2text:
            page = title2text[title]
        elif title in title2file:
            with open(os.path.join(wiki_dir, title2file[title])) as f:
                texts = json.load(f)
                for passage in texts:
                    # temp fix
                    if str(passage["title"]) == "nan":
                        passage["title"] = "NaN"
                    if passage["title"] not in title2text:
                        title2text[passage["title"]] = []
                    title2text[passage["title"]].append(passage["contents"])
            page = title2text[title]
        else:
            if qtype == "long":
                discarded += 1
                continue
            else:
                d["gold_passages"] = []
                d["long_answer_start_end_characters"] = []
                new_data.append(d)
                continue

        bleu_threshold = 0.5
        if qtype == "short" or qtype == "medium":
            answers = [normalize_answer(a) for a in d["answers"]]
            valid_p = []
            for p in page:
                np = normalize_answer(p)
                if any([a in np for a in answers]):
                    valid_p.append(p)
            page = valid_p
            bleu_threshold = 0.0

        gold_passages = []
        long_answer_se = []
        for long_answer in d["long_answers"]:
            paragraph, start_character, end_character, bleu_score = match_answer(tokenizer, long_answer, page, bleu_threshold=bleu_threshold)
            if paragraph is not None:
                gold_passages.append(paragraph)
                long_answer_se.append([start_character, end_character])

        if len(gold_passages) == 0 and qtype == "long":
            discarded += 1
            continue
        #d["long_answers"] = new_long_answers
        d["gold_passages"] = gold_passages
        d["long_answer_start_end_characters"] = long_answer_se

        new_data.append(d)

    return new_data, discarded

def replace_all(input_file, output_file, wiki_dir, num_shards=1):
    print(f"Loading data from {input_file}")
    with open(input_file) as f:
        data = json.load(f).pop("data")
        #data = [d for d in data if d["question_type"] == "long"]

    with open(os.path.join(wiki_dir, "title2file.json")) as f:
        title2file = json.load(f)

    process_examples = partial(replace_examples, title2file, wiki_dir)

    start_time = time.time()
    if num_shards == 1:
        new_data, total_discard = process_examples(data)
    else:
        shards = [[] for _ in range(num_shards)]
        for i, d in enumerate(data):
            shards[i%num_shards].append(d)

        total_discard = 0
        new_data = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for data, discarded in executor.map(process_examples, shards):
                total_discard += discarded
                new_data += data

    end_time = time.time()
    print(f"Finished replacing all long answers, took {end_time - start_time:.2f} seconds, discarded {total_discard} (long answers only) examples.")

    output = {"data": new_data}

    stats = {}
    for d in new_data:
        if d["question_type"] not in stats:
            stats[d["question_type"]] = 0
        stats[d["question_type"]] += 1
    print(stats)

    print(f"Writing data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace the long answers using bleu score similar to KILT.")
    parser.add_argument("--input_file", type=str, default=None,
            help="The input path where the dataset is stored")
    parser.add_argument("--output_file", type=str, default=None,
            help="The output path where the preprocessed dataset will be stored.")
    parser.add_argument("--wiki_dir", type=str, default=None,
            help="The json file containing all the wikipedia passages (can be downloaded from DPR).")
    parser.add_argument("--num_shards", type=int, default=1,
            help="The number of shards to run concurrently (how many tasks we divide into). This is limited by the number of cpus that you have.")
    args = parser.parse_args()

    replace_all(args.input_file, args.output_file, args.wiki_dir, args.num_shards)
