import pandas as pd
import shutil
import os
import csv
import argparse
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim import corpora
from tqdm import tqdm
import pickle

MALLET_PATH = r'/Applications/mallet-2.0.8/bin/mallet'


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
    help="path to models",
)
parser.add_argument(
    "-k",
    "--k_topics",
    type=str,
    help="string of k sizes, like: '1,2,45'",
)
parser.add_argument(
    "-p",
    "--picked",
    type=bool,
    help="is k picked?, True or False",
)


def get_df(input_dir):
    return pd.read_csv(input_dir+"pro_articles_df.csv")


def get_dictionary(input_dir):
    dictionary = Dictionary.load(input_dir+'dictionary')
    return dictionary


def get_corpus(input_dir):
    return corpora.MmCorpus(input_dir+'corpus.mm')

        
def get_ks(k_txt):
    ks = k_txt.split(",")
    ks = [int(k.strip()) for k in ks]
    return ks


def move_files(model, k, output_dir):
    ''' 
    Mallet saves important files needed to reload
    saved models later on in the temp folder. So, 
    we move them out of temp and put them into 
    the models folder of the project dir.
    '''
    files = {f"mallet{k}.thetas": model.fdoctopics(), 
             f"mallet{k}.state": model.fstate()}
    
    for new_path, old_path in files.items():
        base = os.path.basename(old_path)
        shutil.move(old_path, output_dir+base)
        os.rename(output_dir+base, output_dir+new_path)


def run_models(ks, dictionary, corpus, picked, output_dir):
    models = []
    
    if picked == "True": 
        iterations=400
    else:
        iterations=100
        
    for k in ks:
        model = LdaMallet(MALLET_PATH,
                          num_topics=k,
                          corpus=corpus, 
                          id2word=dictionary,
                          iterations=iterations,
                          random_seed=72)
        
        print("Saving model...")
        models.append(model)
        pickle.dump(model, open(output_dir+f"mallet{k}model.pkl", 'wb'))
        move_files(model, k, output_dir)
        
        print("Saving top terms...")
        with open(output_dir+f"top_terms{k}.csv", 'w') as f: 
            writer = csv.writer(f)
            topic_tpls = model.show_topics(num_topics=k, 
                                           num_words=25, 
                                           formatted=False)
            for topic_tpl in topic_tpls: 
                topic_n = topic_tpl[0]
                topic_terms = [term[0] for term in topic_tpl[1]]
                row = [topic_n]
                row.extend(topic_terms)
                writer.writerow(row)
        print()       
    return models
        
        
def main():
    args = parser.parse_args()
    
    df = get_df(args.input_dir)
    print(u'\u2713', "Data Loaded!")

    dictionary = get_dictionary(args.input_dir)
    print(u'\u2713', "Dictionary loaded!")
    
    corpus = get_corpus(args.input_dir)
    print(u'\u2713', "Corpus loaded!")
    
    ks = get_ks(args.k_topics)
    if args.picked == "True": 
        assert len(ks) == 1
        
    print("Initializing MALLET...\n")
    models = run_models(ks, dictionary, corpus, args.picked, args.output_dir)
    print(u'\u2713', "All models trained!")
    
    print("All Done~")
    
if __name__ == "__main__":
    main()