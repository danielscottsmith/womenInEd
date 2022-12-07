import glob
import pandas as pd
import argparse
from tqdm import tqdm
import json


parser = argparse.ArgumentParser(description="Writes an interim df containing articles with standardized col names")
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="path to raw data",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to interim data",
)


def read_raw_data_json(input_dir):
    raw_articles = []
    json_files = glob.glob(input_dir+"*.jsonl")
    for json_file in json_files:
        with open(json_file) as f: 
            for line in tqdm(f): 
                raw_articles.append(json.loads(line))
    raw_articles_df = pd.DataFrame.from_dict(raw_articles)
    return raw_articles_df


def clean_df(df, output_dir):
    # research_df = df[df['docSubType']=='research-article']
    all_articles = []
    for _, article in tqdm(df.iterrows()):
        article_dct = {'journal': article['isPartOf'],
                       'year': article['publicationYear'], 
                       'volume': article['volumeNumber'],
                       'issue': article['issueNumber'],
                       'type': article['docSubType'],
                       'authors': article['creator'],
                       'title': article['title'],
                       'nwords': article['wordCount'],
                       'npages': article['pageCount'],
                       'text': article['fullText']}

        if type(article['creator']) is float: 
            article_dct['nauthors'] = None
            article_dct['auth1_name'] = None
        else:
            article_dct['nauthors'] = len(article['creator'])
            try: 
                article_dct['auth1_name'] = article['creator'][0].split().pop(0)
            except: 
                article_dct['auth1_name'] = None
        all_articles.append(article_dct)
    cleaned_df = pd.DataFrame.from_dict(all_articles)
    cleaned_df.to_feather(output_dir+"interim_articles_df.feather")
    cleaned_df.to_csv(output_dir+"interim_articles_df.csv", index=False)
    
def main():
    args = parser.parse_args()
    
    print("Reading jsonl file...")
    raw_df = read_raw_data_json(args.input_dir)
    print(u'\u2713', "Raw data loaded!\n")
    
    print("Cleaning data...")
    clean_df(raw_df, args.output_dir)
    print(u'\u2713', "Data cleaned!\n")
    print("All Done~")
    
if __name__ == "__main__":
    main()