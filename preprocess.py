import os
import sys
import glob
import threading
import avro
import json
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader
from pyspark.sql.avro.functions import from_avro
from pyspark.sql.types import *
from pyspark.sql.functions import *
import seaborn as sns
from hdf5_getters import *
import pyspark
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F


def correlation_heatmap(corrmatrix, columns):
    # Heatmap produces NaN when values don't vary between them
    # annot = True to showcase values in each cell
    sns.heatmap(corrmatrix, xticklabels=columns, yticklabels=columns, annot=True)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize = 8)
    plt.xticks(rotation=70, fontsize=8)
    plt.title('Attribute Correlation MSD', fontsize=20)
    plt.show()


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

    #   TODO check what to do with file song_hotttnesss. For the time being omit that
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

# Results pretty much the same as the non-scaled version. Now however only the important and non-correlated features should exist
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

def array_splitter(arr):
    # print(arr[-3:])
    # Initial approach with 3 elements from max and 3 from min produced highly correlated results
    return arr[:3] + [0] * (3 - len(arr[:3])) + arr[-3:] + [0] * (3 - len(arr[-3:]))



# Main Function
if __name__ == "__main__":
    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark Preprocessing').master('local[*]').getOrCreate()

    sparkContext = sc.sparkContext
    sparkContext.setLogLevel("OFF")

    parquetFile = sc.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetTimeBig")

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    parquetFile.printSchema()
    parquetFile.show(2, True, True)
    print("Sanity check counter ", parquetFile.count())
    print(len(parquetFile.columns))

    # correlation_checker(parquetFile)

    # LEGACY -> UDF method fetching multiple elements from array features
    pad_fix_length = F.udf(
        lambda arr: array_splitter(arr),
        ArrayType(FloatType())
    )

    # Initial approach. Trying to fetch multiple min, max values per array attribute. Too many correlated elements, try different approach
    # parquetFile = parquetFile.withColumn('segments_loudness_max_norm', pad_fix_length(F.sort_array(col('beats_confidence'), asc=False)))



    parquetFile = parquetFile.withColumn('bars_confidence_max', F.array_max(col('bars_confidence')))
    parquetFile = parquetFile.withColumn('bars_start_max', F.array_max(col('bars_start')))
    parquetFile = parquetFile.withColumn('beats_confidence_max', F.array_max(col('beats_confidence')))
    parquetFile = parquetFile.withColumn('segments_confidence_max', F.array_max(col('segments_confidence')))
    parquetFile = parquetFile.withColumn('segments_loudness_max_time_max',
                                         F.array_max(col('segments_loudness_max_time')))
    parquetFile = parquetFile.withColumn('tatums_confidence_max', F.array_max(col('tatums_confidence')))

    parquetFile = parquetFile.withColumn('bars_confidence_min', F.array_min(col('bars_confidence')))
    parquetFile = parquetFile.withColumn('bars_start_min', F.array_min(col('bars_start')))
    parquetFile = parquetFile.withColumn('beats_confidence_min', F.array_min(col('beats_confidence')))
    parquetFile = parquetFile.withColumn('segments_confidence_min', F.array_min(col('segments_confidence')))
    parquetFile = parquetFile.withColumn('segments_loudness_max_time_min',
                                         F.array_min(col('segments_loudness_max_time')))
    parquetFile = parquetFile.withColumn('tatums_confidence_min', F.array_min(col('tatums_confidence')))

    feature_selector = parquetFile.select("artist_familiarity", "end_of_fade_in", "tempo",
                                          "time_signature_confidence",
                                          "artist_playmeid", "artist_7digitalid", "release_7digitalid",
                                          "key", "loudness", "mode",
                                          "mode_confidence", "time_signature", "label",
                                          "bars_confidence_max", "beats_confidence_max", "bars_start_max",
                                           "segments_confidence_max", "segments_loudness_max_time_max",
                                           "tatums_confidence_max",
                                          "bars_confidence_min",
                                         "beats_confidence_min", "bars_start_min",
                                           "segments_confidence_min", "segments_loudness_max_time_min",
                                          "tatums_confidence_min")



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
                                          "mode_confidence", "time_signature", "label",
                                          "bars_confidence_max", "beats_confidence_max", "bars_start_max",
                                           "segments_confidence_max", "segments_loudness_max_time_max",
                                           "tatums_confidence_max",
                                          "bars_confidence_min",
                                         "beats_confidence_min", "bars_start_min",
                                           "segments_confidence_min", "segments_loudness_max_time_min",
                                          "tatums_confidence_min"]



    assembler = VectorAssembler(inputCols=columns, outputCol="raw_features").setHandleInvalid("skip")

    df_scale = assembler.transform(feature_selector).select('label','raw_features')
    # Most classifiers use some form of a distance calculation and each numeric feature tends to have different
    # ranges, some more broad than others. Scaling these features helps ensure that each feature’s contribution is
    # weighted proportionally.
    # https://albertdchiu.medium.com/a-step-by-step-example-in-binary-classification-5dac0f1ba2dd
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="scaled_features")
    scalerModel = scaler.fit(df_scale)
    df_scale = scalerModel.transform(df_scale).select('label','scaled_features').persist(pyspark.StorageLevel.DISK_ONLY)


    print("Sanity check counter ", df_scale.count())

    df_scale.write.mode("overwrite").parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetAfterProcess")

    df_scale.select('*').show(20, False)

    # correlation_scaled_checker(df_scale)



