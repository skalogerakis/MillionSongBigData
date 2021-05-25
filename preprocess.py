import os
import sys
import glob
import threading
import json
from pyspark.sql.types import *
from pyspark.sql.functions import *
import seaborn as sns
from hdf5_getters import *
import pyspark
import argparse
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspark.ml.feature import VectorAssembler
# import FeatureHasher
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F


'''
    Showcases the final heatmap using seaborn lib
'''
def correlation_heatmap(corrmatrix, columns):
    # Heatmap produces NaN when values don't vary between them
    # annot = True to showcase values in each cell
    sns.heatmap(corrmatrix, xticklabels=columns, yticklabels=columns, annot=True)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize = 8)
    plt.xticks(rotation=70, fontsize=8)
    plt.title('Attribute Correlation MSD', fontsize=20)
    plt.show()


'''
    Initial correlation check for only the numerical fields.
'''
def correlation_checker(parquetFile):
    #
    # feature_selector = parquetFile.select('artist_familiarity', 'artist_hotttnesss', 'artist_latitude',
    # 'artist_longitude', 'song_hotttnesss', 'analysis_sample_rate', 'danceability','duration', 'end_of_fade_in',
    # 'energy', 'key_confidence', 'start_of_fade_out', 'tempo', 'time_signature_confidence', 'artist_playmeid',
    # 'artist_7digitalid', 'release_7digitalid', 'track_7digitalid', 'key', 'mode', 'time_signature', 'year', 'label')

    # parquetFile.select("segments_loudness_max").show(10, False)

    '''
    Try 1: Attributes artist_latitude and artist_longitude seem to be very sparse so they it require to skip a lot of values.
        However they seem irrelevant with the year prediction of a song, so omit them
    Try 2: After examining the first results in our subset, values analysis_sample_rate, danceability, energy contain always the same value
        So the correlation map, they get NaN values which is expected. Omit them as well
    '''


    feature_selector = parquetFile.select('artist_familiarity', 'artist_hotttnesss', 'song_hotttnesss', 'duration',
                                          'end_of_fade_in',
                                          'key_confidence', 'start_of_fade_out', 'tempo', 'time_signature_confidence',
                                          'artist_playmeid',
                                          'artist_7digitalid', 'release_7digitalid', 'track_7digitalid', 'key',
                                          'loudness', 'mode',
                                          'mode_confidence', 'time_signature',
                                          'year', 'label')

    feature_selector.describe().show()

    columns = ['artist_familiarity', 'artist_hotttnesss', 'song_hotttnesss', 'duration',
               'end_of_fade_in',
               'key_confidence', 'start_of_fade_out', 'tempo', 'time_signature_confidence',
               'artist_playmeid',
               'artist_7digitalid', 'release_7digitalid', 'track_7digitalid', 'key', 'loudness', 'mode',
               'mode_confidence', 'time_signature',
               'year', 'label']

    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=columns,
                                outputCol=vector_col).setHandleInvalid("skip")
    corr_vector = assembler.transform(feature_selector).select(vector_col)
    # matrix = Correlation.corr(myGraph_vector, vector_col)

    matrix = Correlation.corr(dataset=corr_vector, column=vector_col, method='pearson').collect()[0][0]
    corrmatrix = matrix.toArray().tolist()
    print(corrmatrix)

    # Check what if scaling causes any differences. This heatmap should showcase only the non-highly correlated elements
    correlation_heatmap(corrmatrix, columns)


'''
    Checks the correlation only on the selected and scaled features. Only the non-correlated features should exist here
'''
def correlation_scaled_checker(scaled_dataset):
    '''
    Try 1: Attributes artist_latitude and artist_longitude seem to be very sparse so they it require to skip a lot of values.
        However they seem irrelevant with the year prediction of a song, so omit them
    Try 2: After examining the first results in our subset, values analysis_sample_rate, danceability, energy contain always the same value
        So the correlation map, they get NaN values which is expected. Omit them as well
    '''
    vector_col = "scaled_features"

    matrix = Correlation.corr(dataset=scaled_dataset, column=vector_col, method='pearson').collect()[0][0]
    corrmatrix = matrix.toArray().tolist()
    print(corrmatrix)

    correlation_heatmap(corrmatrix, columns)


'''
    LEGACY: Part of str_cleaner function. From documentation it is mentioned that some string entries where symbols, so in order not 
    to come up with that problem use this function and in a case like this, replace those entries with empty string(omitted afterwards)
'''


def contains_non_ascii(cur_str):
    if (cur_str.isascii()):
        return cur_str
    else:
        # Non-ascii characters, simply return an empty element which will be omitted later
        return ""


'''
    LEGACY: Part of str_cleaner function. As we will transform string into numbers, we don't really care about human readable part. Also in some
    cases strings could contain more spaces by mistake.
'''


def space_remover(cur_str):
    return cur_str.replace(" ", "")


'''
    LEGACY: Part of str_cleaner function. Convert everything in lower_case so that we wont lose information from capitalized words
'''


def to_lower_case(cur_str):
    return cur_str.lower()


'''
    LEGACY: Part of str_cleaner function. Remove here all the special characters contained in a string
'''


def remove_special(cur_str):
    str_fin = ''.join(e for e in cur_str if e.isalnum())
    return str_fin


'''
    LEGACY: Used in the initial approach while using string fields. These fields where in many cases 'dirty' and could not possibly be used.
    This method performed simple cleaning on strings including checking for non-ascii characters, removing special characters, convert everything to lower case
    and remove spaces
'''


def str_cleaner(cur_str):
    cur_str = contains_non_ascii(cur_str)
    # print(cur_str)
    cur_str = remove_special(cur_str)
    # print(cur_str)
    cur_str = to_lower_case(cur_str)
    # print(cur_str)
    cur_str = space_remover(cur_str)
    # print(cur_str)
    return cur_str

'''
    LEGACY: Used this method to obtain 3 max and min elements from 1D features. Too much correlation between them, 
    simply take max and min in the final implementation
'''
def array_splitter(arr):
    # print(arr[-3:])
    # Initial approach with 3 elements from max and 3 from min produced highly correlated results
    return arr[:3] + [0] * (3 - len(arr[:3])) + arr[-3:] + [0] * (3 - len(arr[-3:]))

'''
    Part of the array_element_column function. Returns the min element of a list(as dataframe entry)
'''
def min_element_column(parquetFiles):
    parquetFiles = parquetFiles.withColumn('bars_confidence_min', F.array_min(col('bars_confidence')))
    parquetFiles = parquetFiles.withColumn('bars_start_min', F.array_min(col('bars_start')))
    parquetFiles = parquetFiles.withColumn('beats_confidence_min', F.array_min(col('beats_confidence')))
    parquetFiles = parquetFiles.withColumn('segments_confidence_min', F.array_min(col('segments_confidence')))
    parquetFiles = parquetFiles.withColumn('segments_loudness_max_time_min',
                                           F.array_min(col('segments_loudness_max_time')))
    parquetFiles = parquetFiles.withColumn('tatums_confidence_min', F.array_min(col('tatums_confidence')))

    return parquetFiles

'''
    Part of the array_element_column function. Returns the max element of a list(as dataframe entry)
'''
def max_element_column(parquetFiles):
    parquetFiles = parquetFiles.withColumn('bars_confidence_max', F.array_max(col('bars_confidence')))
    parquetFiles = parquetFiles.withColumn('bars_start_max', F.array_max(col('bars_start')))
    parquetFiles = parquetFiles.withColumn('beats_confidence_max', F.array_max(col('beats_confidence')))
    parquetFiles = parquetFiles.withColumn('segments_confidence_max', F.array_max(col('segments_confidence')))
    parquetFiles = parquetFiles.withColumn('segments_loudness_max_time_max',
                                           F.array_max(col('segments_loudness_max_time')))
    parquetFiles = parquetFiles.withColumn('tatums_confidence_max', F.array_max(col('tatums_confidence')))

    return parquetFiles

'''
    The following function simply returns from the desired available 1D array fields(the non-highly correlated)
    the max and min element to use as information and train our models in ML
'''
def array_element_column(parquetFiler):
    parquetFiler = max_element_column(parquetFiler)
    parquetFiler = min_element_column(parquetFiler)
    return parquetFiler


# Main Function
if __name__ == "__main__":
    # Need to add the following arguments to execute
    parser = argparse.ArgumentParser(description='This is the preprocessing step app')

    parser.add_argument('--input', help='Requires file input full path')
    parser.add_argument('--output', help='Requires file output full path')
    args = parser.parse_args()

    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark Preprocessing').config("spark.driver.memory", "9g").master(
        'local[*]').getOrCreate()
    #
    sparkContext = sc.sparkContext
    sparkContext.setLogLevel("ERROR")

    parquetFile = sc.read.parquet(str(args.input))

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    parquetFile.printSchema()
    parquetFile.show(2, True, True)
    print("Sanity check counter ", parquetFile.count())
    print(len(parquetFile.columns))

    # Uncomment the following line to check the correlation between numerical fields
    # correlation_checker(parquetFile)

    '''
       LEGACY: The following part is not used in the final implementation. Basically, use the existing string fields
       and convert them into numeric ones for ML analysis. Afterwards, found out that there is not very much valuable info 
       for our model, so omit that  
    '''
    # string_cleaner_udf = F.udf(lambda cur_str: str_cleaner(name), StringType())
    #
    # df_parquetFile = parquetFile.withColumn('artist_id_clean', string_cleaner_udf(col('artist_id')))
    # df_parquetFile = parquetFile.withColumn('artist_location_clean', string_cleaner_udf(col('artist_location')))
    # df_parquetFile = parquetFile.withColumn('artist_mbid_clean', string_cleaner_udf(col('artist_mbid')))
    # df_parquetFile = parquetFile.withColumn('artist_name_clean', string_cleaner_udf(col('artist_name')))
    # df_parquetFile = parquetFile.withColumn('audio_md5_clean', string_cleaner_udf(col('audio_md5')))
    # df_parquetFile = parquetFile.withColumn('release', string_cleaner_udf(col('release')))
    # df_parquetFile = parquetFile.withColumn('song_id', string_cleaner_udf(col('song_id')))
    # df_parquetFile = parquetFile.withColumn('title', string_cleaner_udf(col('title')))
    # df_parquetFile = parquetFile.withColumn('track_id', string_cleaner_udf(col('artist_mbid')))
    #
    #
    # # dataset = dfParquet.select(['artist_name', 'title', 'release', 'artist_id', 'song_id', 'track_id', 'artist_location'])
    #
    # str_column = ['artist_id_clean', 'artist_location_clean', 'artist_mbid_clean', 'artist_name_clean', 'audio_md5_clean',
    #               'release', 'song_id', 'title', 'track_id']
    #
    # # Turn strings into numbers, so that we can use that in ML analysis
    # hasher = FeatureHasher(inputCols=str_column,outputCol="string_features")
    # df_str = hasher.transform(df_parquetFile)

    # LEGACY -> UDF method fetching multiple elements from array features
    pad_fix_length = F.udf(
        lambda arr: array_splitter(arr),
        ArrayType(FloatType())
    )

    # Initial approach. Trying to fetch multiple min, max values per array attribute. Too many correlated elements, try different approach
    # parquetFile = parquetFile.withColumn('segments_loudness_max_norm', pad_fix_length(F.sort_array(col('beats_confidence'), asc=False)))

    # Method to obtain the max and min values of array elements
    parquetFile = array_element_column(parquetFiler=parquetFile)

    # Select the final features and unify them in an array using VectorAssembler
    feature_selector = parquetFile.select("artist_familiarity", "end_of_fade_in", "tempo",
                                          "time_signature_confidence", "artist_playmeid", "artist_7digitalid",
                                          "release_7digitalid",
                                          "key", "loudness", "mode", "mode_confidence", "time_signature",
                                          "bars_confidence_max", "beats_confidence_max", "bars_start_max",
                                          "segments_confidence_max", "segments_loudness_max_time_max",
                                          "tatums_confidence_max", "bars_confidence_min", "beats_confidence_min",
                                          "bars_start_min",
                                          "segments_confidence_min", "segments_loudness_max_time_min",
                                          "tatums_confidence_min", "label")

    # feature_selector = parquetFile.select("artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
    #                                       "time_signature_confidence",
    #                                       "artist_playmeid", "artist_7digitalid", "release_7digitalid",
    #                                       "track_7digitalid", "key", "loudness", "mode",
    #                                       "mode_confidence", "time_signature", "label")

    feature_selector.show(1)
    feature_selector.describe().show()

    # columns = ["artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
    #            "time_signature_confidence",
    #            "artist_playmeid", "artist_7digitalid", "release_7digitalid",
    #            "track_7digitalid", "key", "loudness", "mode",
    #            "mode_confidence", "time_signature", "label"]

    columns = ["artist_familiarity", "end_of_fade_in", "tempo",
               "time_signature_confidence",
               "artist_playmeid", "artist_7digitalid", "release_7digitalid",
               "key", "loudness", "mode",
               "mode_confidence", "time_signature",
               "bars_confidence_max", "beats_confidence_max", "bars_start_max",
               "segments_confidence_max", "segments_loudness_max_time_max",
               "tatums_confidence_max",
               "bars_confidence_min",
               "beats_confidence_min", "bars_start_min",
               "segments_confidence_min", "segments_loudness_max_time_min",
               "tatums_confidence_min"]

    assembler = VectorAssembler(inputCols=columns, outputCol="raw_features").setHandleInvalid("skip")

    df_scale = assembler.transform(feature_selector).select('label', 'raw_features')
    # Most classifiers use some form of a distance calculation and each numeric feature tends to have different
    # ranges, some more broad than others. Scaling these features helps ensure that each feature’s contribution is
    # weighted proportionally.
    # https://albertdchiu.medium.com/a-step-by-step-example-in-binary-classification-5dac0f1ba2dd
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="scaled_features")
    scalerModel = scaler.fit(df_scale)
    df_scale = scalerModel.transform(df_scale).select('label', 'scaled_features').persist(
        pyspark.StorageLevel.DISK_ONLY)

    print("\n\nSanity check counter ", df_scale.count())
    total_count = df_scale.count()
    zero_counter = df_scale.filter(col('label') == 0).count()
    ones_counter = df_scale.filter(col('label') == 1).count()
    print("Count 1s :", ones_counter)
    print("Count 0s", zero_counter)
    print("Sanity check sum 1s and 0s", zero_counter + ones_counter)

    # Not the best weight method. The
    # if(zero_counter > ones_counter):
    #     print("More zeros!")
    #     balance_ratio= float(zero_counter)/float(total_count)
    #     print('BalancingRatio = {}'.format(balance_ratio))
    #     df_scale = df_scale.withColumn("classWeights", when(col('label') == 1, balance_ratio).otherwise(1 - balance_ratio))
    #
    # else:
    #     print("More ones!")
    #     balance_ratio = float(ones_counter) / float(total_count)
    #     print('BalancingRatio = {}'.format(balance_ratio))
    #     df_scale = df_scale.withColumn("classWeights", when(col('label') == 0, balance_ratio).otherwise(1 - balance_ratio))

    # Weights
    # C represent the number of available classes. In our case binary classification, so we have two classes
    c = 2
    weight_fraud = total_count / (c * ones_counter)
    weight_no_fraud = total_count / (c * (total_count - ones_counter))

    df_scale = df_scale.withColumn("classWeights", when(col("label") == 1, weight_fraud).otherwise(weight_no_fraud))

    df_scale.write.mode("overwrite").parquet(str(args.output))

    df_scale.select('*').show(20, False)

    # Uncomment the following line to check the correlation of the final features
    # correlation_scaled_checker(df_scale)
