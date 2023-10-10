import pandas as pd
from datasets import load_dataset, load_dataset_builder
import json
import logging
import argparse


def build_sff():
    """
    Builds and returns the dataset builder for sff
    """
    ds_builder = load_dataset_builder("openai/summarize_from_feedback", "comparisons")
    logging.warning("Building dataset...")
    ds = ds_builder.download_and_prepare()
    logging.warning("Done building dataset...")
    logging.warning(ds_builder.info.description)
    logging.warning(ds_builder.info.features)
    return ds


def load_sff(sample=True, train=True) -> pd.DataFrame:
    """
    Based on the sample and train flags, import sff dataset with the specified
    split and return processed questions and labels dataframes.
    """
    if sample and train:
        ds = load_dataset("openai/summarize_from_feedback", "comparisons", 
                          split='train[0:5000]')
    elif train:
        ds = load_dataset("openai/summarize_from_feedback", "comparisons", 
                          split='train')
    elif sample and not train:
        ds = load_dataset("openai/summarize_from_feedback", "comparisons", 
                          split='validation[0:5000]')
    else:
        ds = load_dataset("openai/summarize_from_feedback", "comparisons", 
                          split='validation')

    df = ds.to_pandas()
    logging.warning('Dataset shape:', df.shape)

    # Data cleaning: dropping unnecessary rows, parsing reddit post column
    df = df.drop(["batch", "split", "extra", "worker"], axis="columns")

    # making subcategories of reddit post columns their own columns
    df['post'] = df['info'].str['post']
    df['title'] = df['info'].str['title']
    df['subreddit'] = df['info'].str['subreddit']

    # drop the original reddit post column and add a question_id and category column 
    # these columns will be used to format the question as outline in LLM_Judge (FastChat)
    df = df.drop("info", axis="columns")
    df["question_id"] = pd.Series(range(df.shape[0]))
    df["category"] = pd.Series((["summary"]*df.shape[0]))

    # aggregating columns into the 'turns' (questions) column
    df['turns'] = df['subreddit'] + " " + df['title'] + " " + df['post']

    # selecting only needed columns
    questions = df[['question_id', 'category', 'turns']].copy()
    questions['turns'] = 'Given the context of the specified subreddit and title, summarize the post. ' + questions['turns']
    questions['turns'] = questions['turns'].apply(lambda x: [x])

    labels = df['choice'].copy()

    return questions, labels


def export_to_json(questions, labels, keep_labels):
    with open("sff_questions.jsonl", "w") as file:
        for item in questions.to_dict('records'):
            json_line = json.dumps(item)
            file.write(json_line + '\n')
    if keep_labels:
        with open("sff_labels.jsonl", "w") as file:
            for item in labels.to_dict('records'):
                json_line = json.dumps(item)
                file.write(json_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add a boolean flag for 'sample' with a default value of False
    parser.add_argument('--sample', action='store_true', default=False,
                        help='Select only a sample of the dataset.')

    # Add a boolean flag for 'train' with a default value of True
    parser.add_argument('--train', action='store_true', default=False,
                        help='Select training split.')

    # Add a boolean flag for 'keep_labels' with a default value of False
    parser.add_argument('--keep_labels', action='store_true', default=False,
                        help='Keep labels.')

    args = parser.parse_args()

    build_sff
    questions, labels = load_sff(args.sample, args.train)
    export_to_json(questions, labels, args.keep_labels)
