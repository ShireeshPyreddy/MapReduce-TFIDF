#!/usr/bin/python
# -*-coding:utf-8 -*

from collections import defaultdict
import subprocess

files = []
words_count_per_doc = {}


def perform_clean(text):
    """
    perform_clean method performs cleaning on all content from all the files.
    :param text: A string on which cleaning is applied.
    :return: modified_text: A string which is cleaned and formatted.
    """
    modified_text = []
    text = text.lower().replace("\n", " ").replace("\t", " ")
    for word in text.split():
        word = word.rstrip(".").rstrip(",")
        if not word.isdigit() and len(word) > 3:
            try:
                float(word)
            except Exception as _:
                modified_text.append(word)

    modified_text = " ".join(modified_text)

    return modified_text


# ---------------------------- Reading the input from hadoop file system using subprocess --------------------------- #
p = subprocess.Popen("hdfs dfs -ls /tfidf_data_5 |  awk '{print $8}'",
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)

for line in p.stdout.readlines():
    if line.strip() and not "doc" in line:
        cat = subprocess.Popen(["hadoop", "fs", "-cat", line.replace("\n", "")], stdout=subprocess.PIPE)
        temp = []
        for line1 in cat.stdout:
            temp.append(perform_clean(line1))
        words_count_per_doc[line.replace("\n", "")] = len(set("\n".join(temp).split()))
        files.append((line.replace("\n", ""), "\n".join(temp)))

# ----------------------------- Performing term frequency using map functionality ----------------------------------- #
map_term_frequency_dict = defaultdict(int)

for each_file in files:
    for each_word in each_file[1].split():
        map_term_frequency_dict[each_file[0] + "#sep#" + each_word] += 1

for map_term_frequency_key, map_term_frequency_value in map_term_frequency_dict.items():
    score = float(map_term_frequency_value) / words_count_per_doc[map_term_frequency_key.split("#sep#")[0]]
    print(map_term_frequency_key, map_term_frequency_value,
          words_count_per_doc[map_term_frequency_key.split("#sep#")[0]], score)
