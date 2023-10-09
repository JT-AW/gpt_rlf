# gpt_rlf
The underlying infrastructure is adopted from FastChat's [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
The goal of this project is to aggregate trustworthy rlhf datasets, generate gpt4 judgements for them, and compare the performance of a reward model trained on gpt4 preference as opposed to human preference (rlhf). 