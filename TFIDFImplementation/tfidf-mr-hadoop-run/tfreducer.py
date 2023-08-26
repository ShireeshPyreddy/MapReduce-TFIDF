#!/usr/bin/python
# -*-coding:utf-8 -*

import math
from collections import defaultdict
import sys

# ------------------ Reading the output from tfmapper.py using sys.stdin and organizing the data -------------------- #
map_term_frequency_dict = {}
for each_input in sys.stdin:
    map_term_frequency_dict[each_input.split(",")[0].lstrip('("').lstrip("('").rstrip("'").rstrip('"')] = float(
        each_input.split(",")[-1].replace(")", ""))

# ------------------ Counting the word occurrence across all the documents ------------------------------------------ #
reduce_inverse_document_frequency_dict = defaultdict(int)
for map_term_frequency_key, _ in map_term_frequency_dict.items():
    reduce_inverse_document_frequency_dict[map_term_frequency_key.split("#sep#")[-1]] += 1


# ------------- Dividing the total documents count by each word frequency and then applying log function ------------ #
idf = {}
for reduce_inverse_document_frequency_key, reduce_inverse_document_frequency_value \
        in reduce_inverse_document_frequency_dict.items():
    idf[reduce_inverse_document_frequency_key] = math.log10(float(5) / reduce_inverse_document_frequency_value)


# ------------- Calculating the TF-IDF by multiplying TF scores with IDF scores ------------------------------------- #
tf_idf = defaultdict(list)

for map_term_frequency_key, map_term_frequency_value in map_term_frequency_dict.items():
    tf_idf[map_term_frequency_key.split("#sep#")[0]].append((map_term_frequency_key.split("#sep#")[1],
                                                           map_term_frequency_value * idf[
                                                               map_term_frequency_key.split("#sep#")[1]]))

for tf_idf_key, tf_idf_value in tf_idf.items():
    for each in tf_idf_value:
        print(tf_idf_key, each[0], each[1])
