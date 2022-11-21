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


# def _fill_bow(unigram_dict):
#     """
#     JSTOR data come as dicts of unigram words as keys
#     with int freqs as values. This helper function 
#     converts the dict into a bag of words by
#     multipling the word by its frequency and 
#     returning a string of unigrams.
#     """
#     all_words = []
#     for word, freq in unigram_dict.items():
#         all_words.append((word + " ") * freq)
#     bow_string = " ".join(all_words)
#     return bow_string


# def _clean_bigrams(bigram_dict):
#     """
#     Helper to preprocess bigrams.
#     """
#     new_dict = {}
#     for bigram, freq in bigram_dict.items():
#         procssed_bigram = preprocess_string(bigram)
#         if len(procssed_bigram)== 2:
#             new_dict[" ".join(procssed_bigram)] = freq
#     return new_dict


# def _count_freq(dct, queries):
#     """
#     Helper to retrieve freqs in preprocessed
#     bigram dict of a list of queries.
#     """
#     i = 0
#     procssed_dct = _clean_bigrams(dct)
#     proccessed_queries = [" ".join(preprocess_string(query)) for query in queries]
#     for query in proccessed_queries:
#         if query in procssed_dct:
#             i+= procssed_dct[query]
#     return i


def read_raw_data_json(input_dir):
    raw_articles = []
    json_files = glob.glob(input_dir+"*.jsonl")
    for json_file in json_files:
        with open(json_file) as f: 
            for line in tqdm(f): 
                raw_articles.append(json.loads(line))
    raw_articles_df = pd.DataFrame.from_dict(raw_articles)
    return raw_articles_df


# def code_method(df):
#     empirical = [
#         "data collection",
#         # "data analysis",
#         # "data methods",
#         # "empirical analysis",
#     ]
    
#     quantitative = [
#         "regression model",
#         # "regression analysis",
#         # "statistical model",
#         # "hypothesis test" ,j
#         # "regression models", 
#         # "statistical models",
#         # "descriptive statistics",
#     ]
#     df['emp'] = df['bigramCount'].apply(lambda x: _count_freq(x, empirical))
#     df['quant'] = df['bigramCount'].apply(lambda x: _count_freq(x, quantitative))
#     return df


def clean_df(df, output_dir):
    research_df = df[df['docSubType']=='research-article']
    all_articles = []
    for _, article in tqdm(research_df.iterrows()):
        article_dct = {'journal': article['isPartOf'],
                       'year': article['publicationYear'], 
                       'volume': article['volumeNumber'],
                       'issue': article['issueNumber'],
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
    
    
def main():
    args = parser.parse_args()
    
    print("Reading jsonl file...")
    raw_df = read_raw_data_json(args.input_dir)
    print(u'\u2713', "Raw data loaded!\n")
    
#     print("Coding article methods...")
#     raw_df = code_method(raw_df)
#     print(u'\u2713', "Methods coded!")
    
    print("Cleaning data...")
    clean_df(raw_df, args.output_dir)
    print(u'\u2713', "Data cleaned!\n")
    print("All Done~")
    
if __name__ == "__main__":
    main()