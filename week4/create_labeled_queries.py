import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import string
import nltk
from nltk.stem import SnowballStemmer

def clean_query(query):
    cleaned_query_lower = query.lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    stemmer = SnowballStemmer("english")

    tokens = []
    for token in cleaned_query_lower.split():
        tokens.append(stemmer.stem(token))

    return ' '.join(tokens)

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--categories", default=categories_file_name, help="the file to output to")
general.add_argument("--queries", default=queries_file_name, help="the file to output to")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output
categories_file_name = args.categories
queries_file_name = args.queries

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# clean the queries
queries = df['query'].transform(clean_query)

# roll up categories
cat_series = df['category']
cat_counts = cat_series.value_counts()
small_cats = cat_counts[cat_counts < min_queries]

while small_cats.size > 0:
    to_roll_up = parents_df.set_index('category')['parent'][small_cats.drop(root_category_id, errors='ignore').index]
    cat_series = cat_series.replace(to_roll_up.to_dict())

    cat_counts = cat_series.value_counts()
    small_cats = cat_counts[cat_counts < min_queries]

# merge our new category series back in
df = df.merge(cat_series.rename('updated_category'), left_index=True, right_index=True)
# merge the cleaned queries in
df = df.merge(queries.rename('cleaned_query'), left_index=True, right_index=True)

# Create labels in fastText format.
df['label'] = '__label__' + df['updated_category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['updated_category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['cleaned_query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
