import hdf5_getters
import csv
import os
import glob


def hd5_info_parser():
    # Open an h5 file in read mode
    h5 = hdf5_getters.open_h5_file_read(
        '/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G/TRAMGDX12903CEF79F.h5')

    function_tracker = filter(lambda x: x.startswith('get'),
                              hdf5_getters.__dict__.keys())  # Detects all the getter functions

    for f in function_tracker:  # Print everything in function tracker
        print(f)

    # First effort to check what each field contains.
    print()  # 55 available fields (exluding number of songs fields)
    print("Num of songs -- ", hdf5_getters.get_num_songs(h5))  # One song per file
    print("Title -- ", hdf5_getters.get_title(h5))  # Print the title of a specific h5 file
    print("Artist familiarity -- ", hdf5_getters.get_artist_familiarity(h5))
    print("Artist hotness -- ", hdf5_getters.get_artist_hotttnesss(h5))
    print("Artist ID -- ", hdf5_getters.get_artist_id(h5))
    print("Artist mbID -- ", hdf5_getters.get_artist_mbid(h5))
    print("Artist playmeid -- ", hdf5_getters.get_artist_playmeid(h5))
    print("Artist 7DigitalID -- ", hdf5_getters.get_artist_7digitalid(h5))
    print("Artist latitude -- ", hdf5_getters.get_artist_latitude(h5))
    print("Artist longitude -- ", hdf5_getters.get_artist_longitude(h5))
    print("Artist location -- ", hdf5_getters.get_artist_location(h5))
    print("Artist Name -- ", hdf5_getters.get_artist_name(h5))
    print("Release -- ", hdf5_getters.get_release(h5))
    print("Release 7DigitalID -- ", hdf5_getters.get_release_7digitalid(h5))
    print("Release -- ", hdf5_getters.get_release(h5))
    print("Song ID -- ", hdf5_getters.get_song_id(h5))
    print("Song Hotness -- ", hdf5_getters.get_song_hotttnesss(h5))
    print("Track 7Digital -- ", hdf5_getters.get_track_7digitalid(h5))
    print("Similar artists -- ", hdf5_getters.get_similar_artists(h5))
    print("Artist terms -- ", hdf5_getters.get_artist_terms(h5))
    print("Artist terms freq -- ", hdf5_getters.get_artist_terms_freq(h5))
    print("Artist terms weight -- ", hdf5_getters.get_artist_terms_weight(h5))
    print("Analysis sample rate -- ", hdf5_getters.get_analysis_sample_rate(h5))
    print("Audio md5 -- ", hdf5_getters.get_audio_md5(h5))
    print("Danceability -- ", hdf5_getters.get_danceability(h5))
    print("Duration -- ", hdf5_getters.get_duration(h5))
    print("End of Fade -- ", hdf5_getters.get_end_of_fade_in(h5))
    print("Energy -- ", hdf5_getters.get_energy(h5))
    print("Key -- ", hdf5_getters.get_key(h5))
    print("Key Confidence -- ", hdf5_getters.get_key_confidence(h5))
    print("Loudness -- ", hdf5_getters.get_loudness(h5))
    print("Mode -- ", hdf5_getters.get_mode(h5))
    print("Mode Confidence -- ", hdf5_getters.get_mode_confidence(h5))
    print("Start of fade out -- ", hdf5_getters.get_start_of_fade_out(h5))
    print("Tempo -- ", hdf5_getters.get_tempo(h5))
    print("Time signature -- ", hdf5_getters.get_time_signature(h5))
    print("Time signature confidence -- ", hdf5_getters.get_time_signature_confidence(h5))
    print("Track ID -- ", hdf5_getters.get_track_id(h5))
    print("Segments Start -- ", hdf5_getters.get_segments_start(h5))
    print("Segments Confidence -- ", hdf5_getters.get_segments_confidence(h5))
    print("Segments Pitches -- ", hdf5_getters.get_segments_pitches(h5))
    print("Segments Timbre -- ", hdf5_getters.get_segments_timbre(h5))
    print("Segments Loudness max -- ", hdf5_getters.get_segments_loudness_max(h5))
    print("Segments Loudness max time-- ", hdf5_getters.get_segments_loudness_max_time(h5))
    print("Segments Loudness start -- ", hdf5_getters.get_segments_loudness_start(h5))
    print("Sections start -- ", hdf5_getters.get_sections_start(h5))
    print("Sections Confidence -- ", hdf5_getters.get_sections_confidence(h5))
    print("Beats start -- ", hdf5_getters.get_beats_start(h5))
    print("Beats confidence -- ", hdf5_getters.get_beats_confidence(h5))
    print("Bars start -- ", hdf5_getters.get_bars_start(h5))
    print("Bars confidence -- ", hdf5_getters.get_bars_confidence(h5))
    print("Tatums start -- ", hdf5_getters.get_tatums_start(h5))
    print("Tatums confidence -- ", hdf5_getters.get_tatums_confidence(h5))
    print("Artist mbtags -- ", hdf5_getters.get_artist_mbtags(h5))
    print("Artist mbtags count -- ", hdf5_getters.get_artist_mbtags_count(h5))
    print("Year -- ", hdf5_getters.get_year(h5))

    fields = ['Title', 'Artist ID']

    with open('Tester.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')

        # writing the fields
        csv_writer.writerow(fields)

        # writing the data rows
        csv_writer.writerow([hdf5_getters.get_title(h5), hdf5_getters.get_artist_id(h5)])

    h5.close()  # close h5 when completed in the end


# Print all titles
def print_all_titles(basedir):
    ext = '.h5'  # Get all files with extension .h5
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        print(files)

        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            print(hdf5_getters.get_title(h5))
            h5.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hd5_info_parser()
    print_all_titles('/home/skalogerakis/Documents/MillionSong/MillionSongSubset/A/M/G/')
