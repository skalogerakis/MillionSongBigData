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

    feature_selector.show(10)
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

    correlation_heatmap(corrmatrix, columns)

# Results pretty much the same as the non-scaled version
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

    feature_selector = parquetFile.select("artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
                                          "time_signature_confidence",
                                          "artist_playmeid", "artist_7digitalid", "release_7digitalid",
                                          "track_7digitalid", "key", "loudness", "mode",
                                          "mode_confidence", "time_signature", "label")



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

    columns = ["artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
                                          "time_signature_confidence",
                                          "artist_playmeid", "artist_7digitalid", "release_7digitalid",
                                          "track_7digitalid", "key", "loudness", "mode",
                                          "mode_confidence", "time_signature"]

    # selected features
    # feature_cols = ['artist_familiarity', 'artist_hotttnesss', 'danceability', 'duration', 'end_of_fade_in', \\\n#            'energy', 'key', 'key_confidence', 'loudness', 'mode', 'mode_confidence', 'song_hotttnesss', \\\n#            'start_of_fade_out', 'tempo', 'time_signature', 'time_signature_confidence', 'year']
    # selected scalar features
    # feature_cols = ['artist_familiarity', 'danceability', 'duration', 'end_of_fade_in',
    #                 'key', 'key_confidence', 'loudness', 'mode',
    #                 'tempo', 'time_signature', 'time_signature_confidence']

    # for i in range(10):
    #     feature_cols.append('segments_loudness_max_time_' + str(i))
    #     feature_cols.append('segments_loudness_max_' + str(i))
    #     feature_cols.append('segments_confidence_' + str(i))
    #     for i in range(120):
    #         feature_cols.append('segments_pitches_' + str(i))
    #         feature_cols.append('segments_timbre_' + str(i))

    assembler = VectorAssembler(inputCols=columns, outputCol="raw_features").setHandleInvalid("skip")

    df_scale = assembler.transform(feature_selector.drop('label'))
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="scaled_features")
    scalerModel = scaler.fit(df_scale)
    df_scale = scalerModel.transform(df_scale).persist(pyspark.StorageLevel.DISK_ONLY)

    print("Sanity check counter ", df_scale.count())

    df_scale.write.mode("overwrite").parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetAfterProcess")

    df_scale.select('*').show(20, False)

    # correlation_scaled_checker(df_scale)



