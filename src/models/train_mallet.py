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
    help="path to dump models & metrics",
)
parser.add_argument(
    "-k",
    "--k_topics",
    type=str,
    help="string of k sizes, like: '1,2,45'",
)


def get_df(input_dir):
    return pd.read_csv(input_dir+"int_articles_df.csv")


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


def run_models(ks, dictionary, corpus, output_dir):
    models = []
    for k in ks: 
        model = LdaMallet(MALLET_PATH,
                          num_topics=k,
                          corpus=corpus, 
                          id2word=dictionary,
                          iterations=100,
                          random_seed=72)
        
        models.append(model)
        model.save(output_dir+f"mallet{k}model")
        move_files(model, k, output_dir)
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
                
    return models


def get_coherence(models, dictionary, corpus, df, output_dir):
    # metrics = ['u_mass', 'c_v', 'c_npmi', 'c_uci']
    coherences = []
    for model in tqdm(models):
        k = len(model.print_topics(num_topics=-1, 
                                   num_words=1))
#         for metric in metrics:
#             if 'c_' in metric: 
#                 cm = CoherenceModel(model=model, 
#                                     corpus=corpus, 
#                                     dictionary=dictionary, 
#                                     coherence=metric, 
#                                     texts=df['tokens'])

#             else:
#                 cm = CoherenceModel(model=model, 
#                                     corpus=corpus, 
#                                     dictionary=dictionary, 
#                                     coherence=metric)
#             coherences.append({'k': k, 
#                                'metric': metric, 
#                                'coherence': cm.get_coherence()})
        cm = CoherenceModel(model=model, 
                            corpus=corpus, 
                            dictionary=dictionary, 
                            coherence='u_mass')
        coherences.append({'k': k, 
                           'metric': 'umass', 
                           'coherence': cm.get_coherence()})

    coherences_df = pd.DataFrame.from_dict(coherences)
    coherences_df.to_csv(output_dir+'coherences_mallet.csv', index=False)
        
        
def main():
    args = parser.parse_args()
    
    df = get_df(args.input_dir)
    print(u'\u2713', "Data Loaded!")

    dictionary = get_dictionary(args.input_dir)
    print(u'\u2713', "Dictionary loaded!")
    
    corpus = get_corpus(args.input_dir)
    print(u'\u2713', "Corpus loaded!")
    
    print("Initializing MALLET...\n")
    ks = get_ks(args.k_topics)
    models = run_models(ks, dictionary, corpus, args.output_dir)
    print(u'\u2713', "All models trained!")
    
    get_coherence(models, dictionary, corpus, df, args.output_dir)
    print(u'\u2713', "Coherence metrics computed!")
    print("All Done~")
    
if __name__ == "__main__":
    main()