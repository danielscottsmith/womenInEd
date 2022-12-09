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
import glob
from statistics import multimode

MALLET_PATH = r'/Applications/mallet-2.0.8/bin/mallet'


parser = argparse.ArgumentParser(description="Fit a series of MALLET LDA models")
parser.add_argument(
    "-i",
    "--in_dir",
    type=str,
    help="path to interim data",
)
parser.add_argument(
    "-m",
    "--model_dir",
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
    type=str,
    help="k been picked: True or False",
)
parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="train eval predict",
)


def get_df(input_dir, task):
    if task == "predict":
        return pd.read_feather(input_dir+"cleaned.feather")
    else:
        df = pd.read_feather(input_dir+"cleaned.feather").sample(frac=.5, random_state=72)
        return df 


def make_dictionary(df, input_dir):
    dictionary = Dictionary(df['tokens'])
    dictionary.filter_extremes(no_above=99)
    dictionary.filter_extremes(no_below=1)
    dictionary.save(input_dir+"dictionary")
    return dictionary


def get_dictionary(input_dir):
    dictionary = Dictionary.load(input_dir+'dictionary')
    return dictionary
    
    
def make_corpus(dictionary, df, input_dir):
    corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]
    # corpora.MmCorpus.serialize(input_dir+'corpus.mm', corpus)
    return corpus
    
        
def get_ks(k_txt):
    ks = k_txt.split(",")
    ks = [int(k.strip()) for k in ks]
    return ks


def _move_files(model, k, model_dir):
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
        shutil.move(old_path, model_dir+base)
        os.rename(model_dir+base, model_dir+new_path)


def train_models(ks, dictionary, corpus, picked, model_dir):
    iterations=100   
    for k in tqdm(ks):
        model = LdaMallet(MALLET_PATH,
                          num_topics=k,
                          corpus=corpus, 
                          id2word=dictionary,
                          iterations=iterations,
                          random_seed=72)
        
        print("Saving model...")
        pickle.dump(model, open(model_dir+f"mallet{k}model.pkl", 'wb'))
        _move_files(model, k, model_dir)
        
        print("Saving top terms...")
        with open(model_dir+f"top_terms{k}.csv", 'w') as f: 
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

        
def get_models(model_dir):
    model_files = glob.glob(model_dir+"*pkl")
    return model_files


def get_coherence(model_files, dictionary, corpus, df, model_dir):
    metrics = ['u_mass', 'c_v', 'c_npmi', 'c_uci']
    with open(model_dir+"topic_coherences.csv", 'a') as f: 
        writer = csv.writer(f)
        writer.writerow(['k', 'metric', 'coherence'])
        for model_file in tqdm(model_files): 
            model = pickle.load(open(model_file, 'rb'))
            k = int(model_file.split('mallet')[1].replace("model.pkl",""))
            for metric in metrics:
                if 'c_' in metric: 
                    cm = CoherenceModel(model=model, 
                                        texts=df['tokens'],
                                        dictionary=dictionary, 
                                        coherence=metric, 
                                        )

                else:
                    cm = CoherenceModel(model=model, 
                                        corpus=corpus, 
                                        dictionary=dictionary, 
                                        coherence=metric)

                coherence_value = cm.get_coherence()
                row = [k, metric, coherence_value]
                writer.writerow(row)
        

def main():
    args = parser.parse_args()
    
    df = get_df(args.in_dir, args.picked)
    print(df.shape)
    print(u'\u2713', "Data Loaded!")
    
    ks = get_ks(args.k_topics)
    
    if args.picked == "True": 
        try: 
            assert len(ks) == 1
        except: 
            print("Select the single k that was picked, else set picked to False.")
        
    if args.task == "train":
        
        dictionary = make_dictionary(df, args.in_dir)
        print(u'\u2713', "Dictionary written!")  
        
        corpus = make_corpus(dictionary, df, args.in_dir)
        print(u'\u2713', "Corpus written!")     
        
        models = train_models(ks, dictionary, corpus, args.picked, args.model_dir)
        print(u'\u2713', "All models trained!")
        print("All Done~")
        
    elif args.task == "eval":
        
        dictionary = get_dictionary(args.in_dir)
        print(u'\u2713', "Dictionary loaded!")  
        
        corpus = make_corpus(dictionary, df, args.in_dir)
        print(u'\u2713', "Corpus written!")    
        
        model_files = get_models(args.model_dir)
        get_coherence(model_files, dictionary, corpus, df, args.model_dir)
        print(u'\u2713', "Coherences computed!")
        print("All Done~")
        
    elif args.task == "predict":
        pass
    
    else: 
        print("Specify 'train', 'eval', or 'predict' as mode.")
        
    
    
if __name__ == "__main__":
    main()