---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
- multiple-choice
- question-answering
language:
- zh
pretty_name: FinEval
size_categories:
- 1K<n<10K
viewer: false
---
<p><h1> The FinEval Dataset </h1></p>

![FinEval Logo](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEvalLogo.jpg "FinEval Logo")

<a name="dataset-announcement"></a>

FinEval is a collection of high-quality multiple-choice questions covering various domains such as finance, economics, accounting, and certifications. It consists of 4,661 questions spanning across 34 distinct academic subjects. To ensure a comprehensive assessment of model performance, FinEval employs various methods including zero-shot, few-shot, answer-only, and chain-of-thought prompts. Evaluating state-of-the-art large language models in both Chinese and English on FinEval reveals that only GPT-4 achieves an accuracy of 60% across different prompt settings, highlighting substantial growth potential of large language models in financial domain knowledge. Our work provides a more comprehensive benchmark for evaluating financial knowledge, utilizing simulated exam data and encompassing a wide range of large language model assessments.

Each subject consists of three splits: dev, val, and test.  The dev set per subject consists of five exemplars with explanations for few-shot evaluation. The val set is intended to be used for hyperparameter tuning. And the test set is for model evaluation. Labels on the test split are not released, users are required to submit their results to automatically obtain test accuracy.

# Language

The language of the data is  Chinese.

# Performance Leaderboard
We divide the evaluation into Answer Only and Chain of Thought. For examples of prompts for both methods, please refer to zero-shot for Answer Only, few-shot for Answer Only, and Chain of Thought.

Below is the average accuracy(%) on the test split. We report the average accuracy over the subjects within each category. "Average" column indicates the average accuracy over all the subjects. Notably, we only report the results from each model under the best setting, which is determined by the highest average accuracy achieved among four settings (i.e., zero- and few-shot learning with and without CoT):

| Model                  | Size    | Finance | Economy | Accounting | Certificate | Average |
|------------------------|---------|:-------:|:-------:|:----------:|:-----------:|:-------:|
| GPT-4                  | unknown |  71.0   |  74.5   |    59.3    |    70.4     |  68.6   |
| ChatGPT                | 175B    |  59.3   |  61.6   |    45.2    |    55.1     |  55.0   |
| Qwen-7B                | 7B      |  54.5   |  54.4   |    50.3    |    55.8     |  53.8   |
| Qwen-Chat-7B           | 7B      |  51.5   |  52.1   |    44.5    |    53.6     |  50.5   |
| Baichuan-13B-Base      | 13B     |  52.6   |  50.2   |    43.4    |    53.5     |  50.1   |
| Baichuan-13B-Chat      | 13B     |  51.6   |  51.1   |    41.7    |    52.8     |  49.4   |
| ChatGLM2-6B            | 6B      |  46.5   |  46.4   |    44.5    |    51.5     |  47.4   |
| InternLM-7B            | 7B      |  49.0   |  49.2   |    40.5    |    49.4     |  47.1   |
| InternLM-Chat-7B       | 7B      |  48.4   |  49.1   |    40.8    |    49.5     |  47.0   |
| LLaMA-2-Chat-70B       | 70B     |  47.1   |  46.7   |    41.5    |    45.7     |  45.2   |
| Falcon-40B             | 40B     |  45.4   |  43.2   |    35.8    |    44.8     |  42.4   |
| Baichuan-7B            | 7B      |  44.9   |  41.5   |    34.9    |    45.6     |  42.0   |
| LLaMA-2-Chat-13B       | 13B     |  41.6   |  38.4   |    34.1    |    42.1     |  39.3   |
| Ziya-LLaMA-13B-v1      | 13B     |  43.3   |  36.9   |    34.3    |    41.2     |  39.3   |
| Bloomz-7b1-mt          | 7B      |  41.4   |  42.1   |    32.5    |    39.7     |  38.8   |
| LLaMA-2-13B            | 13B     |  39.5   |  38.6   |    31.6    |    39.6     |  37.4   |
| ChatGLM-6B             | 6B      |  38.8   |  36.2   |    33.8    |    39.1     |  37.2   |
| Chinese-Llama-2-7B     | 7B      |  37.8   |  37.8   |    31.4    |    36.7     |  35.9   |
| Chinese-Alpaca-Plus-7B | 7B      |  30.5   |  33.4   |    32.7    |    38.5     |  34.0   |
| moss-moon-003-sft      | 16B     |  35.6   |  34.3   |    28.7    |    35.6     |  33.7   |
| LLaMA-2-Chat-7B        | 7B      |  35.6   |  31.8   |    31.9    |    34.0     |  33.5   |
| LLaMA-2-7B             | 7B      |  34.9   |  36.4   |    31.4    |    31.6     |  33.4   |
| AquilaChat-7B          | 7B      |  34.2   |  31.3   |    29.8    |    36.2     |  33.1   |
| moss-moon-003-base     | 16B     |  32.2   |  33.1   |    29.2    |    30.7     |  31.2   |
| Aquila-7B              | 7B      |  27.1   |  31.6   |    32.4    |    33.6     |  31.2   |
| LLaMA-13B              | 13B     |  33.1   |  29.7   |    27.2    |    33.6     |  31.1   |
| Falcon-7B              | 7B      |  28.5   |  28.2   |    27.5    |    27.4     |  27.9   |

# Load the data
```python
from datasets import load_dataset
dataset=load_dataset(r"SUFE-AIFLM-Lab/FinEval",name="finance")

```

Please cite our paper if you use our dataset.
```
@misc{2308.09975,
Author = {Liwen Zhang and Weige Cai and Zhaowei Liu and Zhi Yang and Wei Dai and Yujie Liao and Qianru Qin and Yifei Li and Xingyu Liu and Zhiqiang Liu and Zhoufan Zhu and Anbo Wu and Xin Guo and Yun Chen},
Title = {FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models},
Year = {2023},
Eprint = {arXiv:2308.09975},
}
```