import os
from collections import defaultdict
import math
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class MapReduceTermFrequency:
    """
    The MapReduceTermFrequency class performs Term Frequency (TF) which is first part of TF-IDF using MapReduce
    algorithm.
    """

    @staticmethod
    def map_term_frequency(files):
        """
        map_term_frequency method performs term frequency for each word in each document by splitting each document into
        words and applying map function which keeps counts of each word.
        :param documents: input documents on which map function is applied.
        :return: map_term_frequency_dict: stores the doc name, word and counts of each word from each document. Here the
                 key is the doc_name and the word joined by the separator #sep#.
                 Data Structure: {doc_name#sep#word: count}
                 Note: #sep# behaves like a separator
        """
        map_term_frequency_dict = defaultdict(int)

        for each_file in files:
            for each_word in each_file[1].split():
                map_term_frequency_dict[each_file[0] + "#sep#" + each_word] += 1

        return map_term_frequency_dict

    @staticmethod
    def reduce_term_frequency(map_term_frequency, words_count):
        """
        reduce_term_frequency method takes input from map_term_frequency method and applies reduce operation which
        calculates the term frequency by dividing each word frequency with the total no of words in that document.
        :param map_term_frequency: A dict which contains each word and it's frequency count in each document.
        :param words_count: Dict which contains total no of words present in each document.
        :return: reduce_term_frequency_array: stores the counts of all the occurrences of each word in a document
                 divided by total no of word in that document.
                 Data Structure: [(word, (doc_name, tf_score))]
        """
        reduce_term_frequency_array = []
        for map_term_frequency_key, map_term_frequency_value in map_term_frequency.items():
            reduce_term_frequency_array.append((map_term_frequency_key.split("#sep#")[-1], (
                map_term_frequency_key.split("#sep#")[0],
                map_term_frequency_value / words_count[map_term_frequency_key.split("#sep#")[0]])))

        return reduce_term_frequency_array


class MapReduceInverseDocumentFrequency:
    """
    The MapReduceInverseDocumentFrequency class performs Inverse Document Frequency (IDF) which is second and final part
    of TF-IDF using MapReduce algorithm.
    """

    @staticmethod
    def map_inverse_document_frequency(reduce_term_frequency_array):
        """
        map_inverse_document_frequency method performs inverse document frequency for each word by counting each word
        occurrence in all the documents combined and applying map function which keeps counts of each word across all
        documents.
        :param reduce_term_frequency_array: Array which contains term frequency scores of each word in each document.
        :return: map_inverse_document_frequency_dict: stores the counts of words occurred from all documents.
                 Data Structure: {word: count}
        """
        map_inverse_document_frequency_dict = defaultdict(int)
        for each_tr in reduce_term_frequency_array:
            map_inverse_document_frequency_dict[each_tr[0]] += 1

        return map_inverse_document_frequency_dict

    @staticmethod
    def reduce_inverse_document_frequency(map_inverse_document_frequency_dict, reduce_term_frequency_array,
                                          docs_count):
        """
        reduce_inverse_document_frequency method performs inverse document frequency by taking input from
        map_inverse_document_frequency method and applies reduce operation which at first calculates the IDF by dividing
        total documents count with each word count and then applies log function. Later TF-IDF is performed by
        multiplying TF scores with IDF scores.
        :param map_inverse_document_frequency_dict: A dict which contains the no of times a word has occurred across all
               the documents.
        :param reduce_term_frequency_array:
        :param docs_count: Total no of documents.
        :return: tf_idf: A dict which contains TF-IDF scores for all words documents wise.
                 Data Structure: {"doc_name": [(word, TF-IDF score)]}
        """
        reduce_inverse_document_frequency_dict = {}
        for reduce_inverse_document_frequency_key, reduce_inverse_document_frequency_value \
                in map_inverse_document_frequency_dict.items():
            reduce_inverse_document_frequency_dict[reduce_inverse_document_frequency_key] = math.log10(
                docs_count / reduce_inverse_document_frequency_value)

        print("  Calculating TF-IDF")
        tf_idf = defaultdict(list)

        for each_rtf in reduce_term_frequency_array:
            tf_idf[each_rtf[-1][0]].append(
                (each_rtf[0], each_rtf[-1][-1] * reduce_inverse_document_frequency_dict[each_rtf[0]]))

        return tf_idf


class PerformTasks:
    """
    This class takes the input path, reads all the content present in each text file, cleans each text file and first
    send the data to MapReduceTermFrequency class to calculate Term Frequency and then to
    MapReduceInverseDocumentFrequency class to calculate Inverse Document Frequency. After calculating both TF and IDF,
    it merges both scores and prepare the final tf-idf dictionary. Later this array is used as search index to search
    the keywords given and ranks the links.
    """

    def __init__(self):
        self.stopwords = open("stopwords", 'r').read().replace(' "', "").split('",')

    def perform_clean(self, text):
        """
        perform_clean method performs cleaning on all content from all the files.
        :param text: A string on which cleaning is applied.
        :return: modified_text: A string which is cleaned and formatted.
        """
        modified_text = []
        text = text.lower().replace("\n", " ").replace("\t", " ")
        for word in text.split():
            word = word.strip(".").strip(",")
            if not word.isnumeric() and len(word) > 3:
                try:
                    float(word)
                except Exception as _:
                    if word not in self.stopwords:
                        modified_text.append(word)

        modified_text = " ".join(modified_text)

        return modified_text

    def process_input(self, path):
        """
        process_input method reads and processes all the files from the given path and appends the data to files
        and word count of each document in words_count_per_doc.
        :param path: input to the folder where files are stored.
        :return: None
        """

        files = []
        words_count_per_doc = {}

        for each in os.listdir(path):
            if each.endswith(".txt"):
                print("Reading ", each)
                data = open(path + each, 'r', encoding="utf8").read()
                if each.startswith("link"):
                    print("  Cleaning ", each)
                    cleaned_data = self.perform_clean(data)
                    print("  Cleaning done on", each)
                    files.append((each, cleaned_data))
                    words_count_per_doc[each] = len(set(cleaned_data.split()))

        return files, words_count_per_doc

    @staticmethod
    def perform_search_and_evaluate_results(search_phrase, tfidf_dict, files_count):
        """
        This method searches the given word or phrase in the TF-IDF array and ranks the links by the score. Later, it
        calculates the precision and mse metrics between the Google Rankings and TF-IDF Rankings.
        :param search_phrase: A word or phrase on which TF-IDF is applied.
        :param tfidf_dict: A dict which contains TF-IDF scores for all words documents wise.
        :param files_count: documents count.
        :return: None
        """

        print("\n")
        print('The keyword to search is "', search_phrase, '"')
        doc_rank = {}

        print('Searching"', search_phrase, '"in TF-IDF matrix.')
        for doc, terms in tfidf_dict.items():
            scores = {term: term[-1] for term in terms if term[0] in search_phrase.lower().split()}
            if scores:
                doc_rank[doc] = sum(scores.values())
                if len(scores.values()) == 3:
                    doc_rank[doc] += 1
            else:
                doc_rank[doc] = 0
        print("Searching done.")
        ranked_links = dict(sorted(doc_rank.items(), key=lambda item: item[1], reverse=True))

        # ---------------------------------- Evaluating the results ----------------------------------- #
        actual = [each for each in range(1, files_count + 1)]
        predicted = [int(each.split(".")[0].strip("link")) for each in ranked_links.keys()]
        print("\n")
        print("Precision Score: ")
        for a, p in zip(actual, predicted):
            print("Link" + str(a), round((a / (a + p) * 100), 2))
        print("\n")
        print("MSE Score: ", sum([(a - p) ** 2 for a, p in zip(actual, predicted)]) / files_count)
        print("\n")

        df = pandas.DataFrame([["link" + str(a), "link" + str(p)] for a, p in zip(actual, predicted)],
                              columns=["Google Rank", "IF-TDF Rank"])

        print(df.head(files_count))

        cm = np.zeros((len(actual), len(predicted)))
        for i in range(len(actual)):
            for j in range(len(predicted)):
                if actual[i] == predicted[j]:
                    cm[i, j] = 1

        viz = sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar=False)
        viz.set(xlabel="TF-IDF Rank", ylabel="Google Rank", title='Google vs TF-IDF')

        plt.show()

    def execute_steps(self, path, keyword_to_search):
        """
        The execute_steps method executes the all the other functions sequentially.
        :param: path: input path to read all the files and it's content.
        :return: None
        """

        files, words_count_per_doc = self.process_input(path)
        print("All files read successfully")
        print("\n")
        print("Performing TF using MapReduce")
        map_reduce_term_frequency_object = MapReduceTermFrequency()
        map_term_frequency_dict = map_reduce_term_frequency_object.map_term_frequency(files)
        print("  TF map done")
        reduce_term_frequency_array = map_reduce_term_frequency_object.reduce_term_frequency(map_term_frequency_dict,
                                                                                             words_count_per_doc)
        print("  TF reduce done")

        print("TF is performed successfully")

        print("Performing IDF using MapReduce")
        map_reduce_inverse_document_frequency_obj = MapReduceInverseDocumentFrequency()
        map_inverse_document_frequency_dict = map_reduce_inverse_document_frequency_obj.map_inverse_document_frequency(
            reduce_term_frequency_array)
        print("  IDF map done")
        tf_idf_dict = map_reduce_inverse_document_frequency_obj.reduce_inverse_document_frequency(
            map_inverse_document_frequency_dict, reduce_term_frequency_array, len(files))
        print("  IDF reduce done")
        print("IDF performed successfully\n")

        print("******************TF IDF Scores (Top 20)********************\n")
        print("WORD\t\t    TF-IDF SCORE")
        buffer = {}
        for tf_idf_key, tf_idf_value in tf_idf_dict.items():
            for tf_idf_value_temp in tf_idf_value:
                buffer[tf_idf_value_temp[0]] = tf_idf_value_temp[-1]

        sorted_buffer = dict(sorted(buffer.items(), key=lambda item: item[1], reverse=True))

        count = 0
        for k, v in sorted_buffer.items():
            if len(k) < 7:
                print(k, "\t\t", v)
            else:
                print(k, "\t", v)
            count += 1
            if count == 20:
                break

        self.perform_search_and_evaluate_results(keyword_to_search, tf_idf_dict, len(files))


if __name__ == '__main__':
    input_path = "data\\"
    search_word = "Shireesh Reddy Pyreddy"
    pt_obj = PerformTasks()
    pt_obj.execute_steps(input_path, search_word)
