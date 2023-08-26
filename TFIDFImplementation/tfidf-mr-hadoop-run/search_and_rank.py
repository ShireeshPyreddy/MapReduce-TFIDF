from collections import defaultdict
import subprocess


# ------- Reading the output which is produced by TF-IDF program and collecting the scores in tfidf_scores dict ----- #
p = subprocess.Popen("hdfs dfs -ls /output_10 |  awk '{print $8}'",
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)

tfidf_scores = defaultdict(list)

for line in p.stdout.readlines():
    if line.strip() and "doc" not in line:
        cat = subprocess.Popen(["hadoop", "fs", "-cat", line.replace("\n", "")], stdout=subprocess.PIPE)
        temp = []
        for line1 in cat.stdout:
            temp = line1.replace("\n", "").strip().split(",")
            tfidf_scores[temp[0].replace("('", "").rstrip("'")].append(
                (temp[1].strip().strip("'"), float(temp[-1].rstrip(")").strip())))

# ---------------------------- Searching the given keyword in TF-IDF Matrix ----------------------------------------- #

search_keyword = "shireesh reddy pyreddy"

print('The keyword to search is ', search_keyword, ".")
print('Searching and ranking', search_keyword, ' in TF-IDF matrix.')

doc_rank = {}

for tfidf_key, tfidf_value in tfidf_scores.items():
    scores = {}
    for each in tfidf_value:
        if each[0] in search_keyword.split():
            scores[each] = each[-1]
    if scores:
        doc_rank[tfidf_key] = sum(scores.values())
        if len(scores.values()) == 3:
            doc_rank[tfidf_key] += 1
    else:
        doc_rank[tfidf_key] = 0

print("Searching and ranking is done.")
print("\n")


# ---------------------------------- Evaluating the results by using precision and MSE ------------------------------ #
actual = []
predicted = []
print("Google Rank, TF-IDF Rank")
for _, sorted_ranks in enumerate(sorted(doc_rank.items(), key=lambda item: item[1], reverse=True)):
    print("link" + str(_ + 1), sorted_ranks[0])
    actual.append(_ + 1)
    predicted.append(int(sorted_ranks[0].rsplit("/", 1)[-1].split(".")[0].strip("link")))
print("\n")
print("Precision Score:")
for a, p in zip(actual, predicted):
    print("Link" + str(a), round((float(a) / (float(a) + float(p)) * 100), 2))
print("\n")
print("MSE Sore: ", sum([(a - p) ** 2 for a, p in zip(actual, predicted)]) / float(5))
print("\n")
