import argparse
import json
import requests
import random
import time
import os

from tqdm import tqdm

url = "https://api.openai.com/v1/completions"
all_output = {}
engine = "code-davinci-002"
stop = "\n"
max_tokens = 300
temperature = 0

headers = {
    "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
    "Content-Type": "application/json",
}

post_data = {
    "model": engine,
    "max_tokens": max_tokens,
    "stop": stop,
    "temperature": temperature,
}

def gpt_request(prompt):
    post_data["prompt"] = prompt
    x = requests.post(url, headers=headers, json=post_data)
    return x.json()


def get_icl_prompt(data, n_per_class, include_passage=False):
    if n_per_class == 0:
        return ""

    data_by_class = {}
    for d in data:
        if d["question_type"] not in data_by_class:
            data_by_class[d["question_type"]] = []
        data_by_class[d["question_type"]].append(d)

    use_natural_long = True
    if include_passage and not use_natural_long:
        # we have two choices: take out all the questions that do not have gold passage (if we are trying to match to a specific snapshot)
        data_by_class = {k: [x for x in v if len(x["gold_passages"]) > 0] for k, v in data_by_class.items()}

    if "yesno" in data_by_class:
        yesno_data = data_by_class["yesno"]
        yes = [d for d in yesno_data if d["yes_no_answer"][0] == "YES"]
        no = [d for d in yesno_data if d["yes_no_answer"][0] == "NO"]
        data_by_class["yesno"] = random.sample(yes, n_per_class//2) + random.sample(no, n_per_class//2)

    data_by_class = {k: random.sample(v, n_per_class) for k, v in data_by_class.items()}
    all_classes = list(data_by_class.keys())

    prompt = ""
    for i in range(n_per_class):
        random.shuffle(all_classes)
        for c in all_classes:
            d = data_by_class[c][i]
            gp = d['gold_passages']
            if c == "yesno":
                answer = d["yes_no_answer"][0]
            elif c == "long":
                se = d["long_answer_start_end_characters"]
                answer = [p[s:e] for p, (s,e) in zip(gp, se)]
                answer = random.choice(answer)
            elif c == "medium":
                answer = d["answers"]
                answer = random.choice(answer)
            else:
                # choose the shortest one for short
                answer = d["answers"]
                ans = answer[0]
                for a in answer:
                    ans = a if len(a) < len(ans) else ans
                answer = ans
            prompt += f"Q: {d['question']}\n\n"

            if include_passage:
                if use_natural_long:
                    prompt += f"Passage: {d['long_answers'][0]}\n\n"
                else:
                    prompt += f"Passage: {gp[0]}\n\n"

            prompt += f"A: {answer}\n\n"
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace the long answers using bleu score similar to KILT.")
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--question_types", type=str, default="short,medium,long,yesno")
    parser.add_argument("--ex_per_class", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--include_passage", action="store_true")
    args = parser.parse_args()

    assert args.ex_per_class >= 0
    if args.ex_per_class == 0:
        args.num_trials = 1
    assert args.num_trials >= 1
    question_types = args.question_types.split(",")

    with open(args.train_data) as f:
        train_data = json.load(f)["data"]
        train_data = [d for d in train_data if d["question_type"] in question_types]

    with open(args.test_data) as f:
        test_data = json.load(f)["data"]
        test_data = [d for d in test_data if d["question_type"] in question_types]

    output_dir = os.path.join("outputs", engine)
    assert os.path.exists(output_dir)

    for i in range(args.num_trials):
        random.seed(i)
        icl_prompt = get_icl_prompt(train_data, args.ex_per_class, args.include_passage)
        print(f"Testing trial {i} with the prompt: \n{icl_prompt}\n")

        output_file = os.path.join(output_dir, f"test_{args.question_types}_{args.ex_per_class}_{args.include_passage}_{i}.json")
        all_output = {}
        if os.path.exists(output_file):
            with open(output_file) as f:
                all_output = json.load(f)

        for i, d in enumerate(tqdm(test_data)):
            id = d["id"]
            if id in all_output:
                continue
            prompt = f"{icl_prompt}Q: {d['question']}\n\n"
            passage = ""

            if args.include_passage:
                prompt += f"Passage:"
                completion = gpt_request(prompt)
                while "choices" not in completion:
                    print("choices not found:", completion)
                    with open(output_file, "w") as f:
                        json.dump(all_output, f)
                    time.sleep(5) #sleep for 5 seconds
                    completion = gpt_request(prompt)
                passage = completion["choices"][0]["text"]
                prompt += f"{passage}\n\n"

            prompt += f"A:"
            completion = gpt_request(prompt)
            while "choices" not in completion:
                print("choices not found:", completion)
                with open(output_file, "w") as f:
                    json.dump(all_output, f)
                time.sleep(5) #sleep for 5 seconds
                completion = gpt_request(prompt)
            text = completion["choices"][0]["text"]
            prompt += f"{text}\n\n"


            all_output[id] = {
                "prompt": prompt,
                "passage": passage,
                "text": text,
                "question_type": d["question_type"]
            }

            if (i + 1) % 100 == 0:
                with open(output_file, "w") as f:
                    json.dump(all_output, f)

        with open(output_file, "w") as f:
            json.dump(all_output, f, indent=4)
