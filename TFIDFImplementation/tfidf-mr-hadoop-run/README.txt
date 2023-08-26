Running the program in Hadoop:

Command:
hadoop jar hadoop-streaming-2.7.13.jar -files tfmapper.py,tfreducer.py -input /tfidf_data_5 - output -/output_10 -mapper ./tfmapper.py -reducer ./tfreducer.py

Usages:
tfmapper.py: This file calculates the term frequency by applying map function.
tfreducer.py: This file takes input from tfmapper.py and calculates inverse document frequency by applying reduce
              function.
search_and_rank.py: This file reads the output from tfreducer.py file, performs the keyword search by using TF-IDF
                    scores and then evaluates the results using precision and MSE.

