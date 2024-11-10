import findspark
findspark.init()
from pyspark.sql.functions import explode, split, col, desc
from pyspark.sql import SparkSession
import nltk
from nltk.corpus import stopwords
import shutil
import os
import string

#nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

output_path_1 = "output_1"
output_path_2 = "output_2"
output_path_3 = "output_3"

# Remove the output directories if they already exist
if os.path.exists(output_path_1):
    shutil.rmtree(output_path_1)
if os.path.exists(output_path_2):
    shutil.rmtree(output_path_2)
if os.path.exists(output_path_3):
    shutil.rmtree(output_path_3)

try:
    spark = SparkSession.builder.appName("WordCountApp").master("local[*]").getOrCreate()
    sc = spark.sparkContext
except Exception as e:
    print(f"An error occurred: {e}")

try:
    text_file = sc.textFile("littlewoman.txt")
    text_file_2 = sc.textFile("pride_and_prejudice.txt")

    # Perform transformations and actions
    count_littlewoman = text_file.flatMap(lambda line: line.translate(str.maketrans("", "", string.punctuation)).lower().split()) \
                      .filter(lambda word: word not in stop_words) \
                      .map(lambda word: (word, 1)) \
                      .reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1],ascending=False)

    count_pride_and_prejudice = text_file_2.flatMap(lambda line: line.translate(str.maketrans("", "", string.punctuation)).lower().split()) \
                         .filter(lambda word: word not in stop_words) \
                         .map(lambda word: (word, 1)) \
                         .reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1],ascending=False)
    
    count_combined = text_file.union(text_file_2).flatMap(lambda line: line.translate(str.maketrans("", "", string.punctuation)).lower().split()) \
                      .filter(lambda word: word not in stop_words) \
                      .map(lambda word: (word, 1)) \
                      .reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1],ascending=False)

    # Save output as directories
    count_littlewoman.coalesce(1).saveAsTextFile(output_path_1)
    count_pride_and_prejudice.coalesce(1).saveAsTextFile(output_path_2)
    count_combined.coalesce(1).saveAsTextFile(output_path_3)

    output1 = count_littlewoman.collect()
    output2 = count_pride_and_prejudice.collect()
    output3 = count_combined.collect()

    for path, output_file in [(output_path_1, "output_1.txt"), (output_path_2, "output_2.txt"), (output_path_3, "output_combined_two_files.txt")]:
        with open(output_file, "w") as outfile:
            for filename in sorted(os.listdir(path)):
                if filename.startswith("part-"):
                    with open(os.path.join(path, filename), "r") as infile:
                        outfile.write(infile.read())


    # # Merge partition files into a single output file
    # with open("output_1.txt", "w") as outfile:
    #     for filename in sorted(os.listdir(output_path_1)):
    #         if filename.startswith("part-"):
    #             with open(os.path.join(output_path_1, filename), "r") as infile:
    #                 outfile.write(infile.read())
                    
    # with open("output_2.txt", "w") as outfile:
    #     for filename in sorted(os.listdir(output_path_2)):
    #         if filename.startswith("part-"):
    #             with open(os.path.join(output_path_2, filename), "r") as infile:
    #                 outfile.write(infile.read())
    
    # with open("output_combined_two_files.txt", "w") as outfile:
    #     for filename in sorted(os.listdir(output_path_3)):
    #         if filename.startswith("part-"):
    #             with open(os.path.join(output_path_3, filename), "r") as infile:
    #                 outfile.write(infile.read())
    
    print("Word count files saved successfully.")
    
    text_df = spark.read.text("output_combined_two_files.txt")

    top_25_words = text_df.limit(25)
    top_25_words.show(25, truncate=False)

except Exception as e:
     print(f"An error occurred: {e}")

    