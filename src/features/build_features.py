import pandas as pd 
import argparse
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser


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
    df = pd.read_feather(input_dir+"interim_articles_df.feather")
    return df


def preprocess_text(df, output_dir):
    df['tokens'] = df['text'].apply(lambda x: preprocess_string(" ".join(x)))
    df.to_feather(output_dir+"pro_articles_df.feather")
    return df


def get_bigrams(df, output_dir):
    bigrams = Phrases(df['tokens'], min_count=50)
    bigrams = Phraser(bigrams)
    for idx in range(len(df['tokens'])):
        for tkn in bigrams[df['tokens'].iloc[idx]]:
            if "_" in tkn:
                df['tokens'].iloc[idx].append(tkn)
    df.to_feather(output_dir+"pro_articles_df.feather")
    return df
    

# def get_bow(df, output_dir):
#     for text in tqdm(df['text']):
#         tkn = preprocess_string(text)
#     df['tokens'] = tkn
#     df.to_csv(output_dir+"pro_articles_df.csv", index=False)
#     return df


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
    
    print("Tokenizing and stemming data...")
    df = preprocess_text(df, args.output_dir)
    print(u'\u2713', "Data preprocessed!")
    
    print("Identifying bigrams....")
    df = get_bigrams(df, args.output_dir)
    print(u'\u2713', "Bigrams identified!")
    
    
#     print("Stemming articles...")
#     df = stem_bow(df, args.output_dir)
#     print(u'\u2713', "Articles stemmed!")

    dictionary = make_dictionary(df, args.output_dir)
    print(u'\u2713', "Dictionary written!")
    
    make_corpus(dictionary, df, args.output_dir)
    print(u'\u2713', "Corpus written!")
    print("All Done~")
    
if __name__ == "__main__":
    main()