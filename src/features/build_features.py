from gensim.corpora import Dictionary
from gensim import corpora
import pandas as pd 
import argparse

parser = argparse.ArgumentParser(description="Fit a series of MALLET LDA models")
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="path to interim data",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to dump models & metrics",
)


def get_df(input_dir):
    return pd.read_csv(input_dir+"int_articles_df.csv")


def make_dictionary(df, output_dir):
    # df['tokens'] = df['tokens'].apply(eval)
    # dictionary = Dictionary(df['tokens'])
    # dictionary.filter_extremes(no_above=99)
    # dictionary.filter_extremes(no_below=1)
    # dictionary.save(output_dir+"dictionary")
    dictionary = Dictionary.load(output_dir+'dictionary')
    return dictionary


def make_corpus(dictionary, df, output_dir):
    corpus = [dictionary.doc2bow(eval(doc)) for doc in df['tokens']]
    corpora.MmCorpus.serialize(output_dir+'corpus.mm', corpus)

            
def main():
    args = parser.parse_args()
    
    df = get_df(args.input_dir)
    print(u'\u2713', "Data loaded!")

    dictionary = make_dictionary(df, args.output_dir)
    print(u'\u2713', "Dictionary written!")
    
    make_corpus(dictionary, df, args.output_dir)
    print(u'\u2713', "Corpus written!")

    
if __name__ == "__main__":
    main()