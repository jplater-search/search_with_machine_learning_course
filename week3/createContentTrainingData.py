import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import string
from nltk.stem import SnowballStemmer
import pandas as pd

def transform_name(product_name):
    product_lower_no_punc = product_name.lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    stemmer = SnowballStemmer("english")

    tokens = []
    for token in product_lower_no_punc.split():
        tokens.append(stemmer.stem(token))

    return ' '.join(tokens)

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

general.add_argument("--category_depth", default=0, type=int, help="The ancestors depth to use when parsing category (default is 0 which means leaf).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

min_products = args.min_products
sample_rate = args.sample_rate
category_depth = args.category_depth

print("Writing results to %s" % output_file)
data = []
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        print("Processing %s" % filename)
        f = os.path.join(directory, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if random.random() > sample_rate:
                continue
            # Check to make sure category name is valid
            if (child.find('name') is not None and child.find('name').text is not None and
                child.find('categoryPath') is not None and len(child.find('categoryPath')) > category_depth and
                child.find('categoryPath')[(category_depth if category_depth > 0 else len(child.find('categoryPath'))) - 1][0].text is not None):
                  # Choose last element in categoryPath as the leaf categoryId
                  cat = child.find('categoryPath')[(category_depth if category_depth > 0 else len(child.find('categoryPath'))) - 1][0].text
                  # Replace newline chars with spaces so fastText doesn't complain
                  name = child.find('name').text.replace('\n', ' ')
                  # add tuple to list
                  data.append(("__label__%s" % (cat), transform_name(name)))

# create a data frame
df = pd.DataFrame.from_records(data, columns=['category', 'name'])

# apply min_products filtering if specified
filtered_df = df.groupby('category').filter(lambda x: len(x) >= min_products) if min_products > 0 else df

# dump to csv without headers
filtered_df.to_csv(output_file, sep=' ', index=False, header=False)