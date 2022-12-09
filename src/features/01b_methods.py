import pandas as pd
import argparse
from tqdm import tqdm
from gensim.parsing.preprocessing import preprocess_string

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--in_dir",
    type=str,
    help="interim data subdirectory",
)



EMP = [
# "data analysis",
"data collection",
# "data methods",
# "empirical analysis",
]

REG = [
"regression model",
# "regression analysis",
# "statistical model",
# "hypothesis test" ,
# "regression models", 
# "statistical models",
# "descriptive statistics",
]

def get_df(input_dir):
    df = pd.read_feather(input_dir+"cleaned.feather")
    return df


def _code_article(tokens, queries):
    freq = 0
    for query in queries: 
        query = "_".join(preprocess_string(query))
        freq += list(tokens).count(query)
    return freq


def code_articles(df, input_dir):
    tqdm.pandas()
    df['emp'] = df['tokens'].progress_apply(lambda tokens: _code_article(tokens, EMP))
    df['reg'] = df['tokens'].progress_apply(lambda tokens: _code_article(tokens, REG))
    df.to_feather(input_dir+"cleaned.feather")


def main():
    args = parser.parse_args()
    print("Loading data...")
    df = get_df(args.in_dir)
    print(u'\u2713', "Data loaded!")
    
    
    print("Encoding 'empirical' & 'regression' methods...")
    code_articles(df, args.in_dir)
    print(u'\u2713', "Methods encoded!")
    
    print("All Done~")
    
if __name__ == "__main__":
    main()