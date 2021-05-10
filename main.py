import os
import sys
import glob
import threading
import avro
import json
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader
from pyspark.sql.avro.functions import from_avro

from hdf5_getters import *
from pyspark.sql import SparkSession


# Create first a function that finds all the available paths for parsing
def complete_file_list(basedir):
    ext = '.h5'  # Get all files with extension .h5
    total_file_list = []  # Create first an empty list
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))  # Glob returns a list of paths matching a pathname pattern

        # Since we have multiple arrays simply concat the and after all iteration the final list will contain all the available paths
        total_file_list += files

    print("Path list length ", len(total_file_list))
    return total_file_list

# Idea 1 without avro. Simply return a list of all the attributes
def read_h5_to_list(filename):
    h5 = open_h5_file_read(filename)
    song_num = get_num_songs(h5)
    print(song_num)

    song_info = []
    # METADATA
    # song_info.append(float(get_artist_familiarity(h5tocopy)))
    # song_info.append(float(get_artist_hotttnesss(h5tocopy)))
    # song_info.append(str(get_artist_id(h5tocopy)))
    # song_info.append(str(get_artist_location(h5tocopy)))
    # song_info.append(get_artist_mbtags(h5tocopy).tolist())
    # song_info.append(str(get_artist_name(h5tocopy)))
    # song_info.append(get_artist_terms(h5tocopy).tolist())
    # song_info.append(get_artist_terms_freq(h5tocopy).tolist())
    # song_info.append(get_artist_terms_weight(h5tocopy).tolist())
    # song_info.append(float(get_danceability(h5tocopy)))
    # song_info.append(float(get_duration(h5tocopy)))
    # song_info.append(float(get_end_of_fade_in(h5tocopy)))
    # song_info.append(float(get_energy(h5tocopy)))
    # song_info.append(float(get_key(h5tocopy)))
    # song_info.append(float(get_key_confidence(h5tocopy)))
    # song_info.append(float(get_loudness(h5tocopy)))
    # song_info.append(float(get_mode(h5tocopy)))
    # song_info.append(float(get_mode_confidence(h5tocopy)))
    # song_info.append(str(get_release(h5tocopy)))
    # song_info.append(get_segments_confidence(h5tocopy).tolist())
    # song_info.append(get_segments_loudness_max(h5tocopy).tolist())
    # song_info.append(get_segments_loudness_max_time(h5tocopy).tolist())
    # song_info.append(get_segments_pitches(h5tocopy).tolist())
    # song_info.append(get_segments_timbre(h5tocopy).tolist())
    # song_info.append(get_similar_artists(h5tocopy).tolist())
    # song_info.append(float(get_artist_hotttnesss(h5tocopy)))
    # song_info.append(str(get_song_id(h5tocopy)))
    # song_info.append(float(get_start_of_fade_out(h5tocopy)))
    # song_info.append(float(get_tempo(h5tocopy)))
    # song_info.append(int(get_time_signature(h5tocopy)))
    # song_info.append(float(get_time_signature_confidence(h5tocopy)))
    # song_info.append(str(get_title(h5tocopy)))
    # song_info.append(str(get_track_id(h5tocopy)))
    # song_info.append(int(get_year(h5tocopy)))

    song_info.append(str(get_title(h5)))
    song_info.append(str(get_artist_familiarity(h5)))
    song_info.append(str(get_artist_hotttnesss(h5)))
    song_info.append(str(get_artist_id(h5)))
    song_info.append(str(get_artist_mbid(h5)))
    song_info.append(str(get_artist_playmeid(h5)))
    song_info.append(str(get_artist_7digitalid(h5)))
    song_info.append(str(get_artist_latitude(h5)))
    song_info.append(str(get_artist_longitude(h5)))
    song_info.append(str(get_artist_location(h5)))
    song_info.append(str(get_artist_name(h5)))
    song_info.append(str(get_release(h5)))
    song_info.append(str(get_release_7digitalid(h5)))
    song_info.append(str(get_song_id(h5)))
    song_info.append(str(get_song_hotttnesss(h5)))
    song_info.append(str(get_track_7digitalid(h5)))
    song_info.append(str(get_analysis_sample_rate(h5)))
    song_info.append(str(get_audio_md5(h5)))
    song_info.append(str(get_danceability(h5)))
    song_info.append(str(get_duration(h5)))
    song_info.append(str(get_end_of_fade_in(h5)))
    song_info.append(str(get_energy(h5)))
    song_info.append(str(get_key(h5)))
    song_info.append(str(get_key_confidence(h5)))
    song_info.append(str(get_loudness(h5)))
    song_info.append(str(get_mode(h5)))
    song_info.append(str(get_mode_confidence(h5)))
    song_info.append(str(get_start_of_fade_out(h5)))
    song_info.append(str(get_tempo(h5)))
    song_info.append(str(get_time_signature(h5)))
    song_info.append(str(get_time_signature_confidence(h5)))
    song_info.append(str(get_track_id(h5)))
    song_info.append(str(get_year(h5)))

    print("Song info length ", len(song_info))
    # result.append(song_info)
    h5.close()
    return song_info

# Idea 2 using avro. First write output to avro. However is this necessary????
def song_entry(filename):
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


# Just a word count sanity test to make sure that pyspark works as expected
if __name__ == "__main__":
    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark Word Count').master('local[*]').config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.1.1").getOrCreate()
    sparkContext = sc.sparkContext
    # sparkContext.setLogLevel("ALL")
    # sc.setLogLevel("INFO")




    # filenames = getListOfFiles('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G/')
    filenames = complete_file_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G')
    # print(filenames)
    # print(len(filenames))
    # result = read_h5_to_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G/TRAMGDX12903CEF79F.h5')
    # print(result)
    # print(len(result))

    # IDEA 1: Parallelize per file using the command below and create initial RDDs
    rdd = sparkContext.parallelize(filenames, 4)
    # rdd.foreach(print)

    # IDEA 1: Read h5 files and return a list of all elements
    rdd1 = rdd.map(lambda x: read_h5_to_list(x))

    print("Num of partitions ", rdd1.getNumPartitions())
    print("Count ", rdd1.count())
    print(rdd1.take(50))


    # schema = ["artist familiarity", "artist hotttnesss", "artist id", "artist location", "artist mbtags",
    #             "artist mbtags count", "artist name", "artist terms", "artist terms freq", "artist terms weight",
    #             "danceability", "duration", "end of fade in", "energy", "key",
    #             "key confidence", "loudness", "mode", "mode confidence", "release",
    #             "segments confidence", "segments loudness max", "segments loudness max time",
    #             "segments pitches", "segments timbre", "similar artists",
    #             "song hotttnesss", "song id", "start of fade out", "tempo", "time signature",
    #             "time signature confidence", "title", "track id", "year"]

    # TODO change and add all elements
    schema = ["artist familiarity", "artist hotttnesss"]

    # Transform to Dataframes from rdds from an existing schema
    df1 = rdd1.toDF(schema)
    # df1 = sc.createDataFrame(rdd1, col_name)
    print(df1.take(3))




    # IDEA 2: Create first avro files after parsing h5 files. Read afterwards. Idea 1 seems to be better
    # print(complete_file_list('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G'))
    # result = song_entry('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G/TRAMGDX12903CEF79F.h5')
    # print(result)
    # print(len(result))
    #
    # usersDF = sc.read.format("avro").load("/home/skalogerakis/Projects/MillionSongBigData/songs.avro")
    # usersDF.printSchema()


