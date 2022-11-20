import pandas as pd 
import argparse
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from nltk.stem.porter import *

parser = argparse.ArgumentParser(description="Stem and then writes dictionary obj and serilized corpus obj")
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
    help="path to processed data",
)


def get_df(input_dir):
    return pd.read_csv(input_dir+"interim_articles_df.csv")


def stem_bow(df, output_dir):
    stemmer = PorterStemmer()
    stems = []
    for text in tqdm(df['bow']):
        tkn = preprocess_string(text)
        stem = [stemmer.stem(token) for token in tkn]
        stems.append(stem)
    df['tokens'] = stems
    df.to_csv(output_dir+"pro_articles_df.csv", index=False)
    return df
    

def make_dictionary(df, output_dir):
    dictionary = Dictionary(df['tokens'])
    dictionary.filter_extremes(no_above=99)
    dictionary.filter_extremes(no_below=1)
    dictionary.save(output_dir+"dictionary")
    return dictionary


def make_corpus(dictionary, df, output_dir):
    corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]
    corpora.MmCorpus.serialize(output_dir+'corpus.mm', corpus)

            
def main():
    args = parser.parse_args()
    
    df = get_df(args.input_dir)
    print(u'\u2713', "Data loaded!")
    
    print("Stemming corpus...")
    df = stem_bow(df, args.output_dir)
    print(u'\u2713', "BOW stemmed!")

    dictionary = make_dictionary(df, args.output_dir)
    print(u'\u2713', "Dictionary written!")
    
    make_corpus(dictionary, df, args.output_dir)
    print(u'\u2713', "Corpus written!")
    print("All Done~")
    
if __name__ == "__main__":
    main()