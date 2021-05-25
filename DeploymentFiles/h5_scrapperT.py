import os
import sys
import glob
import threading
# import avro
import json
import argparse
# from avro.datafile import DataFileWriter, DataFileReader
# from avro.io import DatumWriter, DatumReader
# from pyspark.sql.avro.functions import from_avro
from pyspark.sql.types import *
import time
from functools import wraps
import hdf5_getters
from hdf5_getters import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *


def getListOfFiles(dirName):

    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)

        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)


    print("Path list length ", len(allFiles))
    return allFiles


# Create first a function that finds all the available paths for parsing
# def complete_file_list(basedir):
#     ext = '.h5'  # Get all files with extension .h5
#     total_file_list = []  # Create first an empty list
#     for root, dirs, files in os.walk(basedir):
#         files = glob.glob(os.path.join(root, '*' + ext))  # Glob returns a list of paths matching a pathname pattern
#
#         # Since we have multiple arrays simply concat the and after all iteration the final list will contain all the
#         # available paths
#         total_file_list += files
#
#     print("Path list length ", len(total_file_list))
#     return total_file_list

def complete_file_list(basedir, rmv):
    ext = '.h5'  # Get all files with extension .h5
    total_file_list = []  # Create first an empty list
    # s3_list = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))  # Glob returns a list of paths matching a pathname pattern
        # print(files)
        for f in range(len(files)):
            # print(f.replace("/home/skalogerakis/","s3://millionsongsk/"))
            files[f] = files[f].replace(rmv, "s3a://millionsongsk/MSD/MillionSong2/A/A/")

            # print(files[f])
            # tst = f.replace("/home/skalogerakis/","s3://millionsongsk/")
            # s3_list += tst
        # s3_list += tst
        # Since we have multiple arrays simply concat the and after all iteration the final list will contain all the
        # available paths
        total_file_list += files

    # print("Path list length ", len(total_file_list))
    # print(total_file_list)
    return total_file_list

# Time decorator to evaluate performance
def time_wrapper(func):
    def measure_time(*args, **kwargs):
        # Start time function
        start_time = time.time()

        # Function excecution and get results
        results = func(*args, **kwargs)

        # Print the results
        print("Processing time: %.2f seconds." % (time.time() - start_time))
        return results

    return measure_time


# Parse hdf5 file and return all song elements as list
def read_h5_to_list(filename):
    h5 = open_h5_file_read(filename)

    song_info = []

    song_info.append(str(get_title(h5)))
    song_info.append(float(get_artist_familiarity(h5)))
    song_info.append(float(get_artist_hotttnesss(h5)))
    song_info.append(str(get_artist_id(h5)))
    song_info.append(str(get_artist_mbid(h5)))
    song_info.append(int(get_artist_playmeid(h5)))
    song_info.append(int(get_artist_7digitalid(h5)))
    song_info.append(float(get_artist_latitude(h5)))
    song_info.append(float(get_artist_longitude(h5)))
    song_info.append(str(get_artist_location(h5)))
    song_info.append(str(get_artist_name(h5)))
    song_info.append(str(get_release(h5)))
    song_info.append(int(get_release_7digitalid(h5)))
    song_info.append(str(get_song_id(h5)))
    song_info.append(float(get_song_hotttnesss(h5)))
    song_info.append(int(get_track_7digitalid(h5)))
    song_info.append(float(get_analysis_sample_rate(h5)))
    song_info.append(str(get_audio_md5(h5)))
    song_info.append(float(get_danceability(h5)))
    song_info.append(float(get_duration(h5)))
    song_info.append(float(get_end_of_fade_in(h5)))
    song_info.append(float(get_energy(h5)))
    song_info.append(int(get_key(h5)))
    song_info.append(float(get_key_confidence(h5)))
    song_info.append(float(get_loudness(h5)))
    song_info.append(int(get_mode(h5)))
    song_info.append(float(get_mode_confidence(h5)))
    song_info.append(float(get_start_of_fade_out(h5)))
    song_info.append(float(get_tempo(h5)))
    song_info.append(int(get_time_signature(h5)))
    song_info.append(float(get_time_signature_confidence(h5)))
    song_info.append(str(get_track_id(h5)))
    song_info.append(int(get_year(h5)))

    # Elements with arrays
    song_info.append(get_artist_mbtags(h5).tolist())
    song_info.append(get_artist_mbtags_count(h5).tolist())
    song_info.append(get_artist_terms(h5).tolist())
    song_info.append(get_artist_terms_freq(h5).tolist())
    song_info.append(get_artist_terms_weight(h5).tolist())
    song_info.append(get_bars_confidence(h5).tolist())
    song_info.append(get_bars_start(h5).tolist())
    song_info.append(get_beats_confidence(h5).tolist())
    song_info.append(get_beats_start(h5).tolist())
    song_info.append(get_sections_confidence(h5).tolist())
    song_info.append(get_sections_start(h5).tolist())
    song_info.append(get_segments_confidence(h5).tolist())
    song_info.append(get_segments_loudness_max(h5).tolist())
    song_info.append(get_segments_loudness_max_time(h5).tolist())
    song_info.append(get_segments_loudness_start(h5).tolist())
    song_info.append(get_segments_pitches(h5).tolist())
    song_info.append(get_segments_start(h5).tolist())
    song_info.append(get_segments_timbre(h5).tolist())
    song_info.append(get_similar_artists(h5).tolist())
    song_info.append(get_tatums_confidence(h5).tolist())
    song_info.append(get_tatums_start(h5).tolist())

    # print("Song info length ", len(song_info))
    # result.append(song_info)
    h5.close()
    return song_info


# Parse hdf5 element and return everything as a tuple. Seems more suitable for instant fit in dataframes
def read_h5_to_tuple(filename):
    h5 = open_h5_file_read(filename)

    song_info = (
        str(get_title(h5)), float(get_artist_familiarity(h5)), float(get_artist_hotttnesss(h5)), str(get_artist_id(h5)),
        str(get_artist_mbid(h5)), int(get_artist_playmeid(h5)),
        int(get_artist_7digitalid(h5)), float(get_artist_latitude(h5)), float(get_artist_longitude(h5)),
        str(get_artist_location(h5)), str(get_artist_name(h5)), str(get_release(h5)), int(get_release_7digitalid(h5)),
        str(get_song_id(h5)), float(get_song_hotttnesss(h5)), int(get_track_7digitalid(h5)),
        float(get_analysis_sample_rate(h5)),
        str(get_audio_md5(h5)), float(get_danceability(h5)), float(get_duration(h5)), float(get_end_of_fade_in(h5)),
        float(get_energy(h5)), int(get_key(h5)), float(get_key_confidence(h5)), float(get_loudness(h5)),
        int(get_mode(h5)), float(get_mode_confidence(h5)), float(get_start_of_fade_out(h5)), float(get_tempo(h5)),
        int(get_time_signature(h5)), float(get_time_signature_confidence(h5)), str(get_track_id(h5)),
        int(get_year(h5)), get_artist_mbtags(h5).tolist(), get_artist_mbtags_count(h5).tolist(),
        get_artist_terms(h5).tolist(),
        get_artist_terms_freq(h5).tolist(), get_artist_terms_weight(h5).tolist(), get_bars_confidence(h5).tolist(),
        get_bars_start(h5).tolist(), get_beats_confidence(h5).tolist(), get_beats_start(h5).tolist(),
        get_sections_confidence(h5).tolist(), get_sections_start(h5).tolist(), get_segments_confidence(h5).tolist(),
        get_segments_loudness_max(h5).tolist(), get_segments_loudness_max_time(h5).tolist(),
        get_segments_loudness_start(h5).tolist(),
        get_segments_pitches(h5).tolist(), get_segments_start(h5).tolist(), get_segments_timbre(h5).tolist(),
        get_similar_artists(h5).tolist(), get_tatums_confidence(h5).tolist(), get_tatums_start(h5).tolist())

    # print("Song info length ", len(song_info))
    # result.append(song_info)
    h5.close()
    return song_info


'''
    LEGACY METHOD
    One of the initial approaches that writes elements instantly to avro files. However, this approach has the downside
    that does not fully utilize the spark framework and must use DataFileWriter which is suboptimal as files are getting larger
'''


def write_h5_to_avro_instant(filename):
    h5 = open_h5_file_read(filename)
    song_num = get_num_songs(h5)
    print(song_num)

    song_info = []

    song_info.append(str(get_title(h5)))
    song_info.append(str(get_artist_familiarity(h5)))

    print("Song info length ", len(song_info))

    schema_parsed = avro.schema.parse(open("schema.avsc", "rb").read())

    # Write data to an avro file
    writer = DataFileWriter(open("songs.avro", "wb"), DatumWriter(), schema_parsed)
    writer.append({"Title": str(get_title(h5)), "Familiarity": str(get_artist_familiarity(h5))})
    writer.close()

    h5.close()

    return song_info


# Implementation with array schema in use. Works correctly however a little difficult to use
@time_wrapper
def runtime_array(sparkContext):
    filenames = complete_file_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M')
    print(len(filenames))

    # IDEA 1: Parallelize per file using the command below and create initial RDDs
    rdd = sparkContext.parallelize(filenames)

    # IDEA 1: Read h5 files and return a list of all elements
    transformed_rdd = rdd.map(lambda x: read_h5_to_list(x))

    print("Num of partitions ", transformed_rdd.getNumPartitions())
    print("Count ", transformed_rdd.count())

    schema = ["title", "artist_familiarity", "artist_hotttnesss", "artist_id", "artist_mbid", "artist_playmeid",
              "artist_7digitalid", "artist_latitude", "artist_longitude", "artist_location", "artist_name",
              "release", "release_7digitalid", "song_id", "song_hotttnesss", "track_7digitalid", "analysis_sample_rate",
              "audio_md5", "danceability", "duration", "end_of_fade_in", "energy", "key", "key_confidence", "loudness",
              "mode", "mode_confidence", "start_of_fade_out", "tempo", "time_signature", "time_signature_confidence",
              "track_id", "year", "artist_mbtags", "artist_mbtags_count", "artist_terms", "artist_terms_freq",
              "artist_terms_weight", "bars_confidence", "bars_start", "beats_confidence", "beats_start",
              "sections_confidence", "sections_start", "segments_confidence", "segments_loudness_max",
              "segments_loudness_max_time", "segments_loudness_start", "segments_pitches", "segments_start",
              "segments_timbre", "similar_artists", "tatums_confidence", "tatums_start"]

    # Transform to Dataframes from rdds from an existing schema
    df1 = transformed_rdd.toDF(schema)

    print(df1.take(3))
    df1.printSchema()
    df1.show(10, True, True)

    filter_label = df1.filter(col('year') != 0).withColumn('label',
                                                           when(col('year') == 0, -1).when(col('year') < 2000,
                                                                                           0).otherwise(1))

    filter_label.write.mode("overwrite").parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetFile")


# def download_s3_folder(bucket_name, s3_folder, local_dir=None):
#     """
#     Download the contents of a folder directory
#     Args:
#         bucket_name: the name of the s3 bucket
#         s3_folder: the folder path in the s3 bucket
#         local_dir: a relative or absolute directory path in the local file system
#     """
#     bucket = s3.Bucket(bucket_name)
#     for obj in bucket.objects.filter(Prefix=s3_folder):
#         target = obj.key if local_dir is None \
#             else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
#         if not os.path.exists(os.path.dirname(target)):
#             os.makedirs(os.path.dirname(target))
#         if obj.key[-1] == '/':
#             continue
#         bucket.download_file(obj.key, target)

# Implementation using StructTypes instead of arrays to define schema. Seems to be a more strictly formalized
# approach so, will use that
@time_wrapper
def runtime_formalized(sparkContext, sc, input_path, output_path, remover):
    filenames = complete_file_list(str(input_path), str(remover))
    # filenames = getListOfFiles(str(input_path))

    # filenames = complete_file_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M')
    # filenames = complete_file_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/')

    # IDEA 1: Parallelize per file using the command below and create initial RDDs
    # sparkContext.setLogLevel("ERROR")
    rdd = sparkContext.parallelize(filenames)

    # IDEA 1: Read h5 files and return a list of all elements
    transformed_rdd = rdd.map(lambda x: read_h5_to_tuple(x)).persist()

    # Print a few stuff for debugging purposes
    print("Num of partitions ", transformed_rdd.getNumPartitions())
    print("Count ", transformed_rdd.count())

    schema = StructType([
        StructField("title", StringType(), True),
        StructField("artist_familiarity", FloatType(), True),
        StructField("artist_hotttnesss", FloatType(), True),
        StructField("artist_id", StringType(), True),
        StructField("artist_mbid", StringType(), True),
        StructField("artist_playmeid", IntegerType(), True),
        StructField("artist_7digitalid", IntegerType(), True),
        StructField("artist_latitude", FloatType(), True),
        StructField("artist_longitude", FloatType(), True),
        StructField("artist_location", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("release", StringType(), True),
        StructField("release_7digitalid", IntegerType(), True),
        StructField("song_id", StringType(), True),
        StructField("song_hotttnesss", FloatType(), True),
        StructField("track_7digitalid", IntegerType(), True),
        StructField("analysis_sample_rate", FloatType(), True),
        StructField("audio_md5", StringType(), True),
        StructField("danceability", FloatType(), True),
        StructField("duration", FloatType(), True),
        StructField("end_of_fade_in", FloatType(), True),
        StructField("energy", FloatType(), True),
        StructField("key", IntegerType(), True),
        StructField("key_confidence", FloatType(), True),
        StructField("loudness", FloatType(), True),
        StructField("mode", IntegerType(), True),
        StructField("mode_confidence", FloatType(), True),
        StructField("start_of_fade_out", FloatType(), True),
        StructField("tempo", FloatType(), True),
        StructField("time_signature", IntegerType(), True),
        StructField("time_signature_confidence", FloatType(), True),
        StructField("track_id", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("artist_mbtags", ArrayType(StringType()), True),
        StructField("artist_mbtags_count", ArrayType(IntegerType()), True),
        StructField("artist_terms", ArrayType(StringType()), True),
        StructField("artist_terms_freq", ArrayType(FloatType()), True),
        StructField("artist_terms_weight", ArrayType(FloatType()), True),
        StructField("bars_confidence", ArrayType(FloatType()), True),
        StructField("bars_start", ArrayType(FloatType()), True),
        StructField("beats_confidence", ArrayType(FloatType()), True),
        StructField("beats_start", ArrayType(FloatType()), True),
        StructField("sections_confidence", ArrayType(FloatType()), True),
        StructField("sections_start", ArrayType(FloatType()), True),
        StructField("segments_confidence", ArrayType(FloatType()), True),
        StructField("segments_loudness_max", ArrayType(FloatType()), True),
        StructField("segments_loudness_max_time", ArrayType(FloatType()), True),
        StructField("segments_loudness_start", ArrayType(FloatType()), True),
        StructField("segments_pitches", ArrayType(ArrayType(FloatType())), True),
        StructField("segments_start", ArrayType(FloatType()), True),
        StructField("segments_timbre", ArrayType(ArrayType(FloatType())), True),
        StructField("similar_artists", ArrayType(StringType()), True),
        StructField("tatums_confidence", ArrayType(FloatType()), True),
        StructField("tatums_start", ArrayType(FloatType()), True),
    ])

    df = sc.createDataFrame(data=transformed_rdd, schema=schema)
    df.printSchema()
    df.show(2, True, True)

    filter_label = df.filter(col('year') != 0).withColumn('label',
                                                          when(col('year') == 0, -1).when(col('year') < 2000,
                                                                                          0).otherwise(1))

    # filter_label.write.mode("overwrite").parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetBigT")
    filter_label.write.mode("overwrite").parquet(str(output_path))


    '''
        # fdf.write.mode("overwrite").parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetTimeBig")
        PARQUET: Processing time: 470.67 seconds.
        File Size: 233,3MB
    
        # fdf.write.mode("overwrite").format("avro").save("/home/skalogerakis/Projects/MillionSongBigData/avroTimeBig.avro")
        Processing time: 511.12 seconds.
        File Size: 400,6MB
    '''



# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is the h5 scrapper App')

    parser.add_argument('--input', help='Requires file input full path')
    parser.add_argument('--output', help='Requires file output full path')
    parser.add_argument('--rmv', help='Requires file output full path')
    # parser.add_argument('--s3folder', help='Requires file output full path')
    # parser.add_argument('--localdir', help='Requires file output full path')
    args = parser.parse_args()


    # create Spark context with necessary configuration

    # To execute avro execution in Pycharm use the SparkSession below
    # sc = SparkSession.builder.appName('PySpark Word Count').master('local[*]').config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.1.1").getOrCreate()
    # sc = SparkSession.builder.appName('PySpark HDF5 File parser').master('local[*]').getOrCreate()

    # import boto3
    #
    # s3 = boto3.resource(
    #     's3')  # assumes credentials & configuration are handled outside python in .aws directory or environment variables









    spark = SparkSession \
        .builder \
        .appName("PySpark HDF5 File parser") \
        .getOrCreate()

    sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    # download_s3_folder(args.bucket, args.s3folder, args.localdir)

    # runtime_array(sparkContext)
    runtime_formalized(sparkContext, spark, args.input, args.output, args.rmv)
