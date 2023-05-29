#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import collections
import concurrent.futures
from dataclasses import dataclass, field
from functools import partial
import json
import logging
import os
import random
import string
import sys
from typing import Optional, Tuple
import time

import datasets
from datasets import load_dataset
from tqdm import tqdm

import torch
import numpy as np
import evaluate
import transformers
from reader.trainer_reader import ReaderTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version#, send_example_telemetry
from transformers.utils.versions import require_version

from utils import (
    drqa_metric_max_over_ground_truths,
    f1_score,
    normalize_answer,
    exact_match_score,
    drqa_exact_match_score,
    drqa_normalize,
)
from reader.model import BertReader, RobertaReader
from reader.dataset import ReaderDataCollator

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.24.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    train_reader_file: Optional[str] = field(default=None, metadata={"help": "The input reader training data file (a text file)."})
    validation_reader_file: Optional[str] = field(default=None, metadata={"help": "The input reader validation data file (a text file)."})
    test_reader_file: Optional[str] = field(default=None, metadata={"help": "The input reader testing data file (a text file)."})
    train_pred_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input reader training data file (a json file or a comma-separated list of json files)."}
    )
    validation_pred_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input prediction validation data file (a json file or a comma-separated list of json files)."}
    )
    test_pred_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input prediction validation data file (a json file or a comma-separated list of json files)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    question_type: str = field(default="1,2,3,4", metadata={"help": "The question type to use."},)
    num_train_passages: int = field(default=25, metadata={"help": "Number of passages per question for training."})
    num_eval_passages: int = field(default=25, metadata={"help": "Number of passages per question for evaluation."})
    num_answers: int = field(default=10, metadata={"help": "Max number of answers per passage."})

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

def postprocess_examples(n_best_size, tokenizer, min_answer_length, max_answer_length, examples, all_start_logits, all_end_logits, all_rank_logits, features):
    all_predictions = collections.OrderedDict()
    softmax = torch.nn.Softmax(dim=-1)
    for example_index, example in enumerate(tqdm(examples, leave=False)):
        # Those are the indices of the features associated to the current example.
        feature = features[example_index]

        input_ids = feature["input_ids"]
        input_text = feature["input_evidences"]
        offset_mapping = feature["offset_mapping"]

        start_logits = torch.tensor(all_start_logits[example_index])
        end_logits = torch.tensor(all_end_logits[example_index])
        rank_logits = torch.tensor(all_rank_logits[example_index])
        M, L = start_logits.size()

        yes_idx = torch.tensor([id.index(tokenizer.yes_token_id) for id in input_ids])
        no_idx = torch.tensor([id.index(tokenizer.no_token_id) for id in input_ids])

        # zero out the question and title (which should be everything before [YES])
        mask = (torch.arange(0, L, dtype=torch.float32).repeat(M, 1) < yes_idx.view(-1, 1)).bool()
        start_logits[mask] = 0
        end_logits[mask] = 0

        start_prob = softmax(start_logits)
        end_prob = softmax(end_logits)
        rank_prob = softmax(rank_logits)

        start_prob[mask] = 0
        end_prob[mask] = 0

        span_prob = torch.bmm(start_prob.view(M, -1, 1), end_prob.view(M, 1, -1))
        span_prob = torch.triu(span_prob)

        length_mask = torch.ones_like(span_prob)
        length_mask = length_mask * torch.triu(length_mask, diagonal=max(0, min_answer_length-1)) - torch.triu(length_mask, diagonal=min(L, max_answer_length))
        span_prob[length_mask != 1] = 0

        total_prob = span_prob.view(M, -1) * rank_prob.view(M, -1)

        best_prob, best_idx = torch.sort(total_prob.view(-1), descending=True)
        best_passage_idx = torch.div(best_idx, L * L, rounding_mode="trunc")
        best_start_idx = torch.div(best_idx % (L * L), L, rounding_mode="trunc")
        best_end_idx = best_idx % L

        prelim_predictions = []
        #for prob, passage_idx, start_idx, end_idx in zip(best_prob, best_passage_idx, best_start_idx, best_end_idx):
        for i in range(n_best_size):
            prob = best_prob[i]
            passage_idx = best_passage_idx[i]
            start_idx = best_start_idx[i]
            end_idx = best_end_idx[i]

            if len(prelim_predictions) >= n_best_size:
                break
            offset = offset_mapping[passage_idx]
            passage = input_text[passage_idx]
            ids = input_ids[passage_idx]

            if ids[start_idx] == tokenizer.yes_token_id:
                prediction = "YES"
            elif ids[start_idx] == tokenizer.no_token_id:
                prediction = "NO"
            else:
                prediction = passage[offset[start_idx][0]:offset[end_idx][1]]

            prelim_predictions.append(
                {
                    "probability": prob.item(),
                    "text": prediction,
                    "evidence": passage,
                }
            )
        all_predictions[feature["example_id"]] = prelim_predictions[0]
    return all_predictions

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_qa", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None and training_args.do_train:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None and training_args.do_eval:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None and training_args.do_predict:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        def load_preds(pred_files):
            all_preds = {}
            for file in pred_files.split(","):
                with open(file) as f:
                    preds = json.load(f)
                if type(preds) == list:
                    # assume DPR format
                    for p in preds:
                        all_preds[p["id"]] = {
                            "title": [[c["title"]] for c in p["ctxs"]],
                            "evidence": [c["text"] for c in p["ctxs"]],
                            "answer": p["answers"],
                        }
                elif type(preds) == dict:
                    # assume DensePhrases format
                    for k, v in preds.items():
                        all_preds[k] = {"title": v["title"], "evidence": v["evidence"], "answer": v["answer"]}
                else:
                    raise Exception("error reading training pred file")
            return all_preds

        def add_predictions(data, example):
            id = example["id"]
            if id in data:
                d = data[id]
                example["evidence"] = d["evidence"]
                example["title"] = d["title"]
                example["answer"] = d["answer"]
            else:
                example["evidence"] = []
                example["title"] = []
                example["answer"] = []
            return example

        with training_args.main_process_first(desc="Filtering question type"):
            raw_datasets = raw_datasets.filter(lambda example: example["question_type"] in data_args.question_type)
            if "train" in raw_datasets:
                raw_datasets["train"] = raw_datasets["train"].filter(
                    lambda example: example["question_type"] != "yesno" or len(example["gold_passages"]) > 0
                )
        print(raw_datasets)

        if data_args.train_pred_file is not None and "train" in raw_datasets:
            logger.info(f"adding train pred files")
            all_preds = load_preds(data_args.train_pred_file)
            fn = partial(add_predictions, all_preds)
            with training_args.main_process_first(desc="train dataset map predictions"):
                raw_datasets["train"] = raw_datasets["train"].map(
                    fn,
                    desc="Adding predictions to the training set",
                )

        if data_args.validation_pred_file is not None and "validation" in raw_datasets:
            logger.info(f"adding validation pred files")
            all_preds = load_preds(data_args.validation_pred_file)
            fn = partial(add_predictions, all_preds)
            with training_args.main_process_first(desc="train dataset map predictions"):
                raw_datasets["validation"] = raw_datasets["validation"].map(
                    fn,
                    desc="Adding predictions to the validation set",
                )

            #logger.info(f"opening {data_args.validation_reader_file}")
            #with open(data_args.validation_reader_file) as f:
                #data = json.load(f)
            #ids = set(raw_datasets["validation"]["id"])
            #data = {k: v for k, v in data.items() if k in ids}
            #fn = partial(add_predictions, data)
            #with training_args.main_process_first(desc="validation dataset map predictions"):
                #raw_datasets["validation"] = raw_datasets["validation"].map(
                    #fn,
                    #desc="Adding predictions to the validation set",
                #)

        if data_args.test_pred_file is not None and "test" in raw_datasets:
            logger.info(f"adding test pred files")
            all_preds = load_preds(data_args.test_pred_file)
            fn = partial(add_predictions, all_preds)
            with training_args.main_process_first(desc="test dataset map predictions"):
                raw_datasets["test"] = raw_datasets["test"].map(
                    fn,
                    desc="Adding predictions to the test set",
                )

            #logger.info(f"opening {data_args.test_reader_file}")
            #with open(data_args.test_reader_file) as f:
                #data = json.load(f)
            #ids = set(raw_datasets["test"]["id"])
            #data = {k: v for k, v in data.items() if k in ids}
            #fn = partial(add_predictions, data)
            #with training_args.main_process_first(desc="test dataset map predictions"):
                #raw_datasets["test"] = raw_datasets["test"].map(
                    #fn,
                    #desc="Adding predictions to the testing set",
                #)

        with training_args.main_process_first(desc="filter data without predictions"):
            raw_datasets = raw_datasets.filter(lambda example: len(example["evidence"]) > 0)
        print(raw_datasets)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    #model = AutoModelForQuestionAnswering.from_pretrained(
        #model_args.model_name_or_path,
        #from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #config=config,
        #cache_dir=model_args.cache_dir,
        #revision=model_args.model_revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    #)
    model = RobertaReader.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )

    # add our new special tokens to the tokenizer, also store ids for later use
    tokenizer.yes_token = "[YES]"
    tokenizer.no_token = "[NO]"
    new_special_tokens = [tokenizer.yes_token, tokenizer.no_token]
    new_special_tokens_str = tokenizer.sep_token + "".join(new_special_tokens) + tokenizer.sep_token
    new_special_tokens_dict = {"additional_special_tokens": new_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(new_special_tokens_dict)
    tokenizer.yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.yes_token)
    tokenizer.no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.no_token)
    model.resize_token_embeddings(len(tokenizer))

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    #context_column_name = "context" if "context" in column_names else column_names[1]
    context_column_name = "evidence" if "evidence" in column_names else column_names[1]
    #answer_column_name = "answers" if "answers" in column_names else column_names[2]
    answer_column_name = "answer" if "answer" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def find_em(answer_text, passage_text, offset, start=0):
        se_pos = []
        p_len = len(offset)

        for i in range(start, p_len):
            for j in range(i, p_len):
                s = passage_text[offset[i][0]:offset[j][1]]
                if len(s) == 0 or s[0] in string.punctuation:
                    break

                norm_s = normalize_answer(s)
                max_diff = 10
                if len(norm_s) == 0 or abs(len(norm_s) - len(answer_text)) > max_diff:
                    continue

                if answer_text[0] != norm_s[0]:
                    break

                if norm_s == answer_text:
                    se_pos.append((i, j))
                    break

        return se_pos

    def tokenize_question_evidence(examples):
        return examples

    def annotate_answer_span(examples):
        # TODO: we only have to do this for the prepare train features so we just replace that
        # we do all the tokenization first, and then do the anntotations.
        # we can only pass in the input ids after the yes token (to exclude the question), and
        # then add the offset back.

        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        questions = examples[question_column_name]
        evidences = examples.pop(context_column_name)
        all_answers = examples[answer_column_name]
        # tokenize our inputs
        all_input_ids = []
        all_attention_mask = []
        all_offset_mapping = []
        all_titles = examples["title"]

        for i, question in enumerate(examples[question_column_name]):
            passages = evidences[i]
            titles = all_titles[i]
            passages = [t[0] + new_special_tokens_str + p for t, p in zip(titles, passages)]
            tokenized_example = tokenizer(
                [question for _ in passages],
                passages,
                truncation="only_second",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_offsets_mapping=True,
                padding="max_length",
            )
            all_input_ids.append(tokenized_example["input_ids"])
            all_attention_mask.append(tokenized_example["attention_mask"])
            all_offset_mapping.append(tokenized_example["offset_mapping"])

        num_positive_passages = 1

        question_types = examples["question_type"] if "question_type" in examples else None
        gold_passages = examples["gold_passages"] if "gold_passages" in examples else None
        long_answer_se = examples["long_answer_start_end_characters"] if "long_answer_start_end_characters" in examples else None

        start_positions = []
        end_positions = []
        all_positive_idx = []

        for i, input_ids in enumerate(all_input_ids):
            evidence = evidences[i]
            answers = all_answers[i]
            start_pos = []
            end_pos = []

            if question_types is not None and question_types[i] == "yesno":
                # for yes/no questions, we take the passage with the highest f1 score
                # with the long answer as the positive passage
                gold_passages = gold_passages[i]
                se_pos = long_answer_se[i]
                long_answers = [p[s:e] for p, (s,e) in zip(gold_passages, se_pos)]

                f1_scores = [
                    drqa_metric_max_over_ground_truths(
                        lambda x, y: f1_score(x, y)[0],
                        passage,
                        long_answers
                    ) for passage in evidence
                ]
                sorted_idx = np.argsort(f1_scores).tolist()
                positive_idx = sorted_idx[-num_positive_passages:]
                negative_idx = sorted_idx[:data_args.num_train_passages-num_positive_passages]

                # manually annotate the start and end of the appropriate special tokens
                if "YES" in answers:
                    for idx in positive_idx:
                        input_id = input_ids[idx]
                        a_idx = input_id.index(tokenizer.yes_token_id)
                        start_pos.append([a_idx] + [max_seq_length for _ in range(data_args.num_answers-1)])
                        end_pos.append([a_idx] + [max_seq_length for _ in range(data_args.num_answers-1)])
                elif "NO" in answers:
                    for idx in positive_idx:
                        input_id = input_ids[idx]
                        a_idx = input_id.index(tokenizer.no_token_id)
                        start_pos.append([a_idx] + [max_seq_length for _ in range(data_args.num_answers-1)])
                        end_pos.append([a_idx] + [max_seq_length for _ in range(data_args.num_answers-1)])
                else:
                    raise Exception("expected YES or NO in answer")

                for idx in negative_idx:
                    start_pos.append([max_seq_length for _ in range(data_args.num_answers)])
                    end_pos.append([max_seq_length for _ in range(data_args.num_answers)])
            else:
                for idx, input_id in enumerate(input_ids):
                    start_pos.append([])
                    end_pos.append([])
                    s_pos = input_id.index(tokenizer.yes_token_id)
                    offsets = all_offset_mapping[i][idx]
                    for a in answers:
                        norm_a = normalize_answer(a)
                        se_pos = find_em(norm_a, evidence[idx], offsets, s_pos)
                        for (s, e) in se_pos:
                            if len(start_pos[-1]) < data_args.num_answers:
                                start_pos[-1].append(s)
                                end_pos[-1].append(e)
                    while len(start_pos[-1]) < data_args.num_answers:
                        start_pos[-1].append(max_seq_length)
                    while len(end_pos[-1]) < data_args.num_answers:
                        end_pos[-1].append(max_seq_length)

                positive_idx = [idx for idx in range(len(start_pos)) if start_pos[idx][0] != max_seq_length]
                negative_idx = [idx for idx in range(len(start_pos)) if start_pos[idx][0] == max_seq_length]

                positive_idx = positive_idx[:num_positive_passages]
                negative_idx = negative_idx[:data_args.num_train_passages-num_positive_passages]

            all_positive_idx.append(positive_idx)
            all_passage_idx = positive_idx + negative_idx
            all_input_ids[i] = [all_input_ids[i][idx] for idx in all_passage_idx]
            all_attention_mask[i] = [all_attention_mask[i][idx] for idx in all_passage_idx]
            all_offset_mapping[i] = [all_offset_mapping[i][idx] for idx in all_passage_idx]
            start_positions.append(start_pos)
            end_positions.append(end_pos)

        # make sure every question has at least num_train_passages
        # use positive passages from other questions if not
        answer_masks = []
        for i, input_ids in enumerate(all_input_ids):
            if len(input_ids) < data_args.num_train_passages:
                question = questions[i]
                remaining = data_args.num_train_passages - len(input_ids)
                candidates = [(all_titles[x][y], evidences[x][y]) for x, p_idx in enumerate(all_positive_idx) for y in p_idx if x != i]
                if remaining < 1:
                    raise Exception("should be >= 1")
                elif remaining > len(candidates):
                    import pdb; pdb.set_trace()
                    raise Exception(f"looked for {remaining} but only have {len(candidates)} candidates")
                passages = random.sample(candidates, remaining)
                passages = [t[0] + new_special_tokens_str + p for t, p in passages]
                tokenized_example = tokenizer(
                    [question for _ in passages],
                    passages,
                    truncation="only_second",
                    max_length=max_seq_length,
                    stride=data_args.doc_stride,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
                all_input_ids[i] += tokenized_example["input_ids"]
                all_attention_mask[i] += tokenized_example["attention_mask"]
                all_offset_mapping[i] += tokenized_example["offset_mapping"]
            answer_masks.append([[1 if p < max_seq_length else 0 for p in pos] for pos in start_positions[i]])

        examples["input_ids"] = all_input_ids
        examples["attention_mask"] = all_attention_mask
        examples["offset_mapping"] = all_offset_mapping
        examples["start_positions"] = start_positions
        examples["end_positions"] = end_positions
        examples["answer_mask"] = answer_masks

        return examples

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        num_positive_passages = 1

        answer_spans = examples["answer_span"]
        question_types = examples["question_type"]
        examples["positive_evidence"] = []
        examples["negative_evidence"] = []
        examples["start_positions"] = []
        examples["end_positions"] = []
        evidences = examples.pop("evidence")

        # annotate each question with positive and negative evidences
        # note that we may or may not end up with a positive evidence for every question
        # so we need to filter out the questions without positive evidence for training set later
        for i, evidences in enumerate(evidences):
            examples["positive_evidence"].append([])
            examples["negative_evidence"].append([])
            examples["start_positions"].append([])
            examples["end_positions"].append([])

            question_type = question_types[i]
            titles = examples["title"][i]
            if question_type == "yesno":
                # for yes/no questions, we take the passage with the highest f1 score
                # with the long answer as the positive passage
                gold_passages = examples["gold_passages"][i]
                se_pos = examples["long_answer_start_end_characters"][i]
                long_answers = [p[s:e] for p, (s,e) in zip(gold_passages, se_pos)]

                f1_scores = [
                    drqa_metric_max_over_ground_truths(
                        lambda x, y: f1_score(x, y)[0],
                        passage,
                        long_answers
                    ) for passage in evidences
                ]
                sorted_idx = np.argsort(f1_scores).tolist()
                positive_idx = sorted_idx[-num_positive_passages:]
                negative_idx = sorted_idx[:data_args.num_train_passages-num_positive_passages]
            else:
                # for other questions, we use the annotated answer spans
                answer_span = answer_spans[i]
                positive_idx = [i for i, s in enumerate(answer_span) if len(s) > 0]
                negative_idx = [i for i, s in enumerate(answer_span) if len(s) == 0]
                negative_idx = negative_idx[:data_args.num_train_passages-num_positive_passages]

                spans = [answer_span[idx] for idx in positive_idx[:num_positive_passages]]
                starts = [[s[0] for s in span] for span in spans]
                ends = [[s[1] for s in span] for span in spans]
                examples["start_positions"][i] += starts
                examples["end_positions"][i] += ends

            # we also prepend the passage with the title and special tokens for tokenization later
            examples["positive_evidence"][i] += [titles[idx][0] + new_special_tokens_str + evidences[idx] for idx in positive_idx]
            examples["negative_evidence"][i] += [titles[idx][0] + new_special_tokens_str + evidences[idx] for idx in negative_idx]

        # use positive passages from other questions as negatives if we don't have enough
        all_positive_passages = [passage for passages in examples["positive_evidence"] for passage in passages]
        for i, negative_evidences in enumerate(examples["negative_evidence"]):
            if len(negative_evidences) < data_args.num_train_passages - num_positive_passages:
                # it's possible to sample one's own positive passage but very unlikely given sufficient batch size
                num_missing = data_args.num_train_passages - num_positive_passages - len(negative_evidences)
                negative_evidences += random.sample(all_positive_passages, num_missing)

        # tokenize our inputs
        examples["input_ids"] = []
        examples["attention_mask"] = []
        examples["offset_mapping"] = []
        examples["answer_mask"] = []
        for i, question in enumerate(examples[question_column_name]):
            question_type = question_types[i]
            answers = examples[answer_column_name][i]
            passages = examples["positive_evidence"][i][:num_positive_passages] + examples["negative_evidence"][i]
            assert len(passages) <= data_args.num_train_passages

            tokenized_example = tokenizer(
                [question for _ in passages],
                passages,
                truncation="only_second",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_offsets_mapping=True,
                padding="max_length",
            )

            examples["input_ids"].append(tokenized_example["input_ids"])
            examples["attention_mask"].append(tokenized_example["attention_mask"])
            examples["offset_mapping"].append(tokenized_example["offset_mapping"])

            if question_type == "yesno":
                # add start and end pos for yes/no questions
                if "YES" in answers:
                    se = tokenized_example["input_ids"][0].index(tokenizer.yes_token_id)
                    examples["start_positions"][i].append([se])
                    examples["end_positions"][i].append([se])
                elif "NO" in answers:
                    se = tokenized_example["input_ids"][0].index(tokenizer.no_token_id)
                    examples["start_positions"][i].append([se])
                    examples["end_positions"][i].append([se])
            else:
                # add offset to previous start and end positions
                for j, (start_positions, end_positions) in enumerate(zip(examples["start_positions"][i], examples["end_positions"][i])):
                    # we expect the [NO] token to be two tokens before the start of the passage
                    offset = tokenized_example["input_ids"][j].index(tokenizer.no_token_id) + 2
                    examples["start_positions"][i][j] = [p + offset for p in start_positions]
                    examples["end_positions"][i][j] = [p + offset for p in end_positions]

            # pad the start and end positions so it's easy to collate the data later
            while len(examples["start_positions"][i]) < data_args.num_train_passages:
                examples["start_positions"][i].append([max_seq_length for _ in range(data_args.num_answers)])
            while len(examples["end_positions"][i]) < data_args.num_train_passages:
                examples["end_positions"][i].append([max_seq_length for _ in range(data_args.num_answers)])
            for positions in examples["start_positions"][i]:
                while len(positions) < data_args.num_answers:
                    positions.append(max_seq_length)
                while len(positions) > data_args.num_answers:
                    positions.pop()
            for positions in examples["end_positions"][i]:
                while len(positions) < data_args.num_answers:
                    positions.append(max_seq_length)
                while len(positions) > data_args.num_answers:
                    positions.pop()
            examples["answer_mask"].append([])
            for positions in examples["start_positions"][i]:
                examples["answer_mask"][i].append([1 if p < max_seq_length else 0 for p in positions])

        return examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            print(f"Selected {max_train_samples} for training: {train_dataset}")
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                annotate_answer_span,
                #prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # only use the questions that have at least one answer
        train_dataset = train_dataset.filter(lambda example: example["answer_mask"][0][0] == 1)
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        question_types = examples["question_type"]
        evidences = examples.pop("evidence")
        examples["input_evidences"] = []

        for i, evidence in enumerate(evidences):
            titles = examples["title"][i]
            examples["input_evidences"].append([titles[idx][0] + new_special_tokens_str + e for idx, e in enumerate(evidence)][:data_args.num_eval_passages])
            while len(examples["input_evidences"][i]) < data_args.num_eval_passages:
                examples["input_evidences"][i].append(new_special_tokens_str)

        # tokenize our inputs
        examples["input_ids"] = []
        examples["attention_mask"] = []
        examples["offset_mapping"] = []
        examples["example_id"] = []
        for i, question in enumerate(examples[question_column_name]):
            question_type = question_types[i]
            answers = examples[answer_column_name]
            passages = examples["input_evidences"][i]
            examples["example_id"].append(examples["id"][i])

            tokenized_example = tokenizer(
                [question for _ in passages],
                passages,
                truncation="only_second",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_offsets_mapping=True,
                padding="max_length",
            )

            examples["input_ids"].append(tokenized_example["input_ids"])
            examples["attention_mask"].append(tokenized_example["attention_mask"])
            examples["offset_mapping"].append(tokenized_example["offset_mapping"])

        return examples

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )
    data_collator = ReaderDataCollator()

    def postprocess_reader_predictions(
        examples,
        features,
        tokenizer,
        predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
        n_best_size: int = 20,
        min_answer_length: int = 0,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        # we don't actually use examples here, features has everything we need due to preprocessing

        if len(predictions) != 3:
            raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, rank_logits).")
        all_start_logits, all_end_logits, all_rank_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        softmax = torch.nn.Softmax(dim=-1)

        postprocess_fn = partial(postprocess_examples, n_best_size, tokenizer, min_answer_length, max_answer_length)
        if len(examples) < 4000:
            all_predictions = postprocess_fn(examples, all_start_logits, all_end_logits, all_rank_logits, features)
        else:
            num_shards = 64
            def get_shards(arr):
                shards = [[] for _ in range(num_shards)]
                for i, a in enumerate(arr):
                    shards[i % num_shards].append(a)
                return shards
            shards = get_shards(examples)
            start_shards = get_shards(all_start_logits)
            end_shards = get_shards(all_end_logits)
            rank_shards = get_shards(all_rank_logits)
            feature_shards = get_shards(features)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for predictions in executor.map(postprocess_fn, shards, start_shards, end_shards, rank_shards, feature_shards):
                    all_predictions.update(predictions)

        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(output_dir, f"predictions_{data_args.question_type}.json")
            logger.info(f"Saving predictions to {prediction_file}")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        return all_predictions

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_reader_predictions(
            examples=examples,
            features=features,
            tokenizer=tokenizer,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        formatted_predictions = [{"id": k, "prediction_text": v["text"]} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        id2answers = {k["id"]: k["answers"] for k in p.label_ids}
        id2predictions = {k["id"]: k["prediction_text"] for k in p.predictions}

        all_em = []
        all_f1 = []

        for k, pred in id2predictions.items():
            answers = id2answers[k]
            em = drqa_metric_max_over_ground_truths(drqa_exact_match_score, pred, answers)
            f1 = drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], pred, answers)
            all_em.append(em)
            all_f1.append(f1)

        output = {
            "exact_match": 100 * sum(all_em) / len(all_em),
            "f1": 100 * sum(all_f1) / len(all_f1),
        }

        #return metric.compute(predictions=p.predictions, references=p.label_ids)
        return output

    # Initialize our Trainer
    trainer = ReaderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        #import pdb; pdb.set_trace()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            trainer.save_model(os.path.join(training_args.output_dir, "best_dev"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if metrics is None:
            metrics = {}
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

