# ☕ MoQA: Multi-type Open-domain Question Answering

Check out our paper [here](https://aclanthology.org/2023.dialdoc-1.2/)!

## Setup

You can set up the virual environment using conda:
```
conda env create -f environment.yml
```

## Data 

You can download the pre-processed data from here:

You can also find all the pre-processing scripts under `./preprocess`.

## Reproducing baselines

### DPR

You can use the original [DPR repo](https://github.com/facebookresearch/DPR) for training DPR. 
We provide the prediction files for the `DPR-All` model.

### Extractive Reader

You can train the extractive reader with:
```
make run-reader
```
You can change the hyperparameters in the Makefile.

### FiD Reader

You can use the original [FiD repo](https://github.com/facebookresearch/FiD) for training FiD.

### LLMs

You must first set the environment variable `OPENAI_API_KEY` to your API key.

For the LLM experiments: 
```
make test-gpt3
```
You can change the model name and other configurations in the Makefile.

### Classifier

```
make run-classifier
```

## Contacts

For any questions, please feel free to reach out to Howard (`hyen@princeton.edu`).

