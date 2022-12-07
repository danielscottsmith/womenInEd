from genderize import Genderize
import pandas as pd 
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser(description="Writes an interim df containing articles with standardized col names")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="processed data dir",
)

API_KEY = 'bbe489da604ffbc66a8b9f0764dd88ce'

def _remove_initials(name):
    if name: 
        new = name.replace(".", "")
        new = new.strip()
        if len(new) < 2:
            return None
        else: 
            return name
    else:
        return None
    
def get_df(data_dir):
    return pd.read_feather(data_dir+"pro_articles_df.feather")


def clean_name(df):
    df['auth1_name'] = df['auth1_name'].apply(lambda x: _remove_initials(x))
    return df


def lookup_gender(df):
    genderize = Genderize(api_key = API_KEY, timeout=60)
    genderized_names = {}
    for name in tqdm(set(df['auth1_name'].unique())):
        if name:
            sex = genderize.get([name])[0]['gender']
            prob_woman = genderize.get([name])[0]['probability']
            if sex == 'female':
                gender = 'woman'
            elif sex == 'male': 
                gender = 'man'
                prob_woman = 1 - prob_woman 
            genderized_names[name] = (gender, prob_woman)
        else: 
            genderized_names[name] = (None, None)      
    return genderized_names


def get_genders(df, genderized_names, data_dir):
    df['auth1_gender'] = df['auth1_name'].apply(lambda x: genderized_names[x][0])
    df['auth1_probwom'] = df['auth1_name'].apply(lambda x: genderized_names[x][1])
    df.to_feather(data_dir+"pro_articles_df.feather")
    return df 


def main():
    args = parser.parse_args()
    df = get_df(args.data_dir)
    print(u'\u2713', "Data loaded!")
    
    df = clean_name(df)
    print(u'\u2713', "First names cleaned!")
    
    genderized_names = lookup_gender(df)
    print(u'\u2713', "Set of names looked up!")
    
    get_genders(df, genderized_names, args.data_dir)
    print(u'\u2713', "Names classified!")
    

if __name__ == "__main__":
    main()
    