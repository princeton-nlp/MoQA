import argparse
import csv
import json
import os

from bert_score import BERTScorer
import numpy as np
import rouge
from tqdm import tqdm
import torch

"""
Use https://github.com/pltrdy/rouge for rouge-l evaluation.
UnifiedQA(https://github.com/allenai/unifiedqa) also uses this library but it appears to be an older version
"""
rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
)

def rouge_l_score(prediction, ground_truths):
    if len(prediction) == 0:
        return 0
    return max([rouge_l_evaluator.get_scores(prediction, g)[0]["rouge-l"]["f"] for g in ground_truths])


def load_bert_preds(pred_path):
    with open(pred_path) as f:
        preds= json.load(f)
    preds = {k: v["text"] for k, v in preds.items()}
    return preds

def load_bert_dir(dir_path):
    all_preds = {}
    for qtype in ["short", "medium", "long", "yesno"]:
        preds = load_bert_preds(os.path.join(dir_path, f"predictions_{qtype}.json"))
        all_preds.update(preds)
    return all_preds

def load_fid_preds(pred_path):
    with open(pred_path) as f:
        lines = csv.reader(f, delimiter="\t")
        preds = {str(l[0]): str(l[1]) for l in lines}
    return preds

def load_fid_dir(dir_path):
    all_preds = {}
    for qtype in ["short", "medium", "long", "yesno"]:
        preds = load_fid_preds(os.path.join(dir_path, f"final_output_test_{qtype}.txt"))
        all_preds.update(preds)
    return all_preds

def load_gpt_dir(data_file):
    with open(data_file) as f:
        preds = json.load(f)
    preds = {k: v["text"] for k, v in preds.items()}
    return preds

def load_classifier(pred_path):
    id2label = {}
    with open(pred_path) as f:
        lines = csv.reader(f, delimiter="\t")
        for i, line in enumerate(lines):
            if i == 0:
                continue
            id2label[line[2]] = line[1]
    return id2label

def load_data(data_file):
    with open(data_file) as f:
        data = json.load(f)["data"]
    all_data = {}
    for d in data:
        id = d["id"]
        qtype = d["question_type"]
        if qtype == "null":
            continue
        elif qtype == "long":
            gold_passages = d["gold_passages"]
            se_pos = d["long_answer_start_end_characters"]
            all_data[id] = {"answers": [p[s:e] for p, (s,e) in zip(gold_passages, se_pos)]}
        elif qtype == "yesno":
            all_data[id] = {"answers": d["yes_no_answer"]}
        else:
            all_data[id] = {"answers": d["answers"]}
        all_data[id]["question_type"] = qtype
        all_data[id]["question"] = d["question"]
    return all_data

def load_and_calculate(data, classes, short_dir, medium_dir, long_dir, yesno_dir, short_format, medium_format, long_format, yesno_format, print_examples, bertscorer=None):
    predictions = {}

    #temp_file = "/n/fs/nlp-hyen/jiw/all_table_ids.json"
    #with open(temp_file) as f:
        #candidate_ids = json.load(f)

    predictions["short"] = load_fid_dir(short_dir) if short_format == "fid" else load_bert_dir(short_dir) if short_format == "bert" else load_gpt_dir(short_dir)
    predictions["medium"] = load_fid_dir(medium_dir) if medium_format == "fid" else load_bert_dir(medium_dir) if medium_format == "bert" else load_gpt_dir(medium_dir)
    predictions["long"] = load_fid_dir(long_dir) if long_format == "fid" else load_bert_dir(long_dir) if long_format == "bert" else load_gpt_dir(long_dir)
    predictions["yesno"] = load_fid_dir(yesno_dir) if yesno_format == "fid" else load_bert_dir(yesno_dir) if yesno_format == "bert" else load_gpt_dir(yesno_dir)

    metrics = ["em", "f1", "rouge-l", "length", "bertscore", "preds", "answers"]
    results = {
        "short": {x: [] for x in metrics},
        "medium": {x: [] for x in metrics},
        "long": {x: [] for x in metrics},
        "yesno": {x: [] for x in metrics},
    }

    count = 0
    for i, (id, d) in enumerate(data.items()):
        #if id not in candidate_ids:
            #continue

        qtype = d["question_type"]
        question = d["question"]
        classification = classes[id]
        #print(f"qtype = {qtype}\nquestion: {question}\nclassification: {classification}")
        #import pdb; pdb.set_trace()
        if args.oracle_classifier:
            classification = qtype

        #if id not in predictions[classification]:
            #continue

        pred = predictions[classification][id]
        answers = list(set(d["answers"]))

        em = drqa_metric_max_over_ground_truths(drqa_exact_match_score, pred, answers)
        if qtype == "short" or qtype == "medium":
            f1 = drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], pred, answers)
            results[qtype]["f1"].append(f1)
        if qtype == "long":
            rouge_l = rouge_l_score(pred, answers)
            results[qtype]["rouge-l"].append(rouge_l)

        results[qtype]["em"].append(em)
        results[qtype]["length"].append(len(pred.split()))
        results[qtype]["preds"].append(pred)
        results[qtype]["answers"].append(answers)

        if print_examples and qtype == "short" and count < 10:
            print("-" * 30)
            print(f"Question Type: {qtype}; Question: {question}\n")
            print(f"Answer: {answers}\n")
            print(f"Prediction: {pred}")
            print("-" * 30)
            count += 1

    if bertscorer is not None:
        with torch.inference_mode():
            for qtype in tqdm(["short", "medium", "long", "yesno"]):
                scores = bertscorer.score(results[qtype]["preds"], results[qtype]["answers"], batch_size=8)
                results[qtype]["bertscore"] = [x.tolist() for x in scores]

    micro_em = []
    macro_em = []
    ret = []

    micro_bert = []
    macro_bert = []
    bert_ret = []

    output = ""
    lengths_output = ""
    bertscore_output = ""
    for qtype in ["short", "medium", "long", "yesno"]:
    #for qtype in ["yesno"]:
        em = results[qtype]["em"]
        f1 = results[qtype]["f1"]
        rouge_l = results[qtype]["rouge-l"]
        micro_em += em
        macro_em.append(100 * sum(em) / len(em) if len(em) > 0 else -1)

        if qtype == "yesno":
            ret.append(macro_em[-1])
            output += f"{ret[-1]:.02f} "
        elif qtype == "long":
            ret.append(macro_em[-1])
            ret.append(100 * sum(rouge_l) / len(rouge_l) if len(rouge_l) > 0 else -1)
            output += f"{ret[-2]:.02f} {ret[-1]:.02f} "
        else:
            ret.append(macro_em[-1])
            ret.append(100 * sum(f1) / len(f1) if len(f1) > 0 else -1)
            output += f"{ret[-2]:.02f} {ret[-1]:.02f} "

        if macro_em[-1] == -1:
            macro_em.pop()

        lengths_output += f"{np.mean(results[qtype]['length']):.1f} ({np.std(results[qtype]['length']):.1f}),"
        if bertscorer is not None:
            scores = np.array(results[qtype]["bertscore"]) * 100
            micro_bert += scores[2].tolist()
            scores = scores.mean(1)
            macro_bert.append(scores[2])
            bert_ret.append(scores[2])
            #bertscore_output += f"{scores[0]:.1f},{scores[1]:.1f},{scores[2]:.1f},"
            bertscore_output += f"{scores[2]:.1f},"

    ret.append(sum(macro_em) / len(macro_em))
    ret.append(100 * sum(micro_em) / len(micro_em))
    output += f"{ret[-2]:.02f} {ret[-1]:.02f}"
    print(output)

    temp = results["short"]['length'] + results["medium"]['length'] + results["long"]['length']
    lengths_output += f"{np.mean(temp):.1f} ({np.std(temp):.1f}),"
    temp += results["yesno"]["length"]
    lengths_output += f"{np.mean(temp):.1f} ({np.std(temp):.1f})"
    #print("lengths: ", lengths_output)

    bert_ret += [np.mean(macro_bert), np.mean(micro_bert)]
    bertscore_output += f"{np.mean(macro_bert):.1f},{np.mean(micro_bert):.1f}"
    print("bertscores:", bertscore_output)

    return bert_ret if bertscorer is not None else ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate with classifier")
    parser.add_argument("--classifier_predictions", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--short_model_dir", type=str, default=None, help="contains all the prediction files")
    parser.add_argument("--short_format", type=str, default=None, help="bert or fid or gpt")
    parser.add_argument("--medium_model_dir", type=str, default=None, help="contains all the prediction files")
    parser.add_argument("--medium_format", type=str, default=None, help="bert or fid or gpt")
    parser.add_argument("--long_model_dir", type=str, default=None, help="contains all the prediction files")
    parser.add_argument("--long_format", type=str, default=None, help="bert or fid or gpt")
    parser.add_argument("--yesno_model_dir", type=str, default=None, help="contains all the prediction files")
    parser.add_argument("--yesno_format", type=str, default=None, help="bert or fid or gpt")
    parser.add_argument("--oracle_classifier", action="store_true", help="assume gold classifier")
    parser.add_argument("--bertscore", action="store_true", help="calculate bertscore")
    parser.add_argument("--print_examples", action="store_true", help="print examples")
    parser.add_argument("--num_trials", type=int, default=0, help="assume seed and iterate over trials if greater than 1")

    args = parser.parse_args()

    data = load_data(args.data_file)
    classes = load_classifier(args.classifier_predictions)

    if args.bertscore:
        model_type = "microsoft/deberta-xlarge-mnli"
        num_layers = 40
        bertscorer = BERTScorer(model_type=model_type, num_layers=num_layers)
        print(f"bertscorer hash: {bertscorer.hash}")
    else:
        bertscorer = None

    print("Short-EM Short-F1 Medium-EM Medium-F1 Long-EM Long-ROUGEL YesNo-Acc MacroAvg MicroAvg")
    if args.num_trials > 0:
        results = []
        for i in range(args.num_trials):
            ret = load_and_calculate(data, classes,
                args.short_model_dir + f"_{i}.json",
                args.medium_model_dir + f"_{i}.json",
                args.long_model_dir + f"_{i}.json",
                args.yesno_model_dir + f"_{i}.json",
                args.short_format,
                args.medium_format,
                args.long_format,
                args.yesno_format,
                args.print_examples,
                bertscorer,
            )
            results.append(ret)
        output = ""
        results = np.array(results)
        for stat in results.transpose():
            output += f"{np.mean(stat):.1f}({np.std(stat):.1f}),"
        print(output)

    else:
        load_and_calculate(data, classes, args.short_model_dir, args.medium_model_dir, args.long_model_dir, args.yesno_model_dir, args.short_format, args.medium_format, args.long_format, args.yesno_format, args.print_examples, bertscorer)
