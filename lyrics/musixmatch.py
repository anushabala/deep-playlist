__author__ = 'anushabala'
from argparse import ArgumentParser
import os
import urllib
import json
import codecs
from datetime import datetime

API_KEYS=['2fabf028b619a95f4f7be3e74c861605',
          '3a437273d2a492605b7fef2d3ac0fcc8',
          '33b5652bfdf12f7a4333fda64b2947ec',
          '859a94723b830c48b6d8d3ec9560a2c5',
          '029b2743a7c0371b8076a8090f999af1']
musiXmatch_endpoint = 'http://api.musixmatch.com/ws/1.1'
lyrics_endpoint = 'track.lyrics.get'
NOT_ENGLISH = "NOT_EN"


def get_lyrics(track_id, api_key):
    params = {'track_id':track_id, 'apikey':api_key}
    url = "%s/%s?%s" % (musiXmatch_endpoint, lyrics_endpoint, urllib.urlencode(params))
    result = json.loads(urllib.urlopen(url).read(), encoding='utf-8')
    # print result
    # print result["message"]["header"]

    status_code = result["message"]["header"]["status_code"]
    # status_code = 200
    # lyrics = "into you, into youuuuu"
    if status_code != 200:
        return status_code, None
    lyrics = result["message"]["body"]["lyrics"]["lyrics_body"]
    if "en" not in result["message"]["body"]["lyrics"]["lyrics_language"]:
        return NOT_ENGLISH, None

    return status_code, lyrics

def get_similar(track_info):
    similars = [s[0] for s in track_info["similars"] if s[1] > 0.05]
    return similars

def load_mappings(mappings_file):
    mappings = {}
    inp = open(mappings_file, 'r')
    for line in inp.readlines():
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        line = line.split("<SEP>")
        mappings[line[0]] = line[3]

    inp.close()
    return mappings


def get_already_loaded_tracks(out_dir):
    tracks = []
    for f in os.listdir(out_dir):
        id = f.replace(".txt", "")
        tracks.append(id)

    return tracks


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str, help='Location of lastFM dataset to get lyrics from')
    parser.add_argument('--out_dir', required=True, type=str,  help='Directory to write lyrics to. If the directory '
                                                                    'already exists, existing files may be '
                                                                    'overwritten!')
    parser.add_argument('--save_progress', type=str, default='./lyrics_scraper.progress', help='File to log progress of scraper to')
    parser.add_argument('--lim', type=int, default=-1, help='Maximum number of songs to request lyrics for')
    parser.add_argument('--mappings_file', type=str, default='./mxm_779k_matches.txt', help='File to load mappings (MSD -> musixmatch) from')
    parser.add_argument('--start_from', type=str, default='none', help='Name of file containing LastFM info for last track that was processed.')
    parser.add_argument('--stop_at', type=str, default='none', help='Name of last file to process')

    args = parser.parse_args()
    start = False
    if args.start_from == 'none':
        start = True
    mappings = load_mappings(args.mappings_file)
    progress_file = open(args.save_progress, 'a')
    curr_api_key = 0
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    loaded_files = get_already_loaded_tracks(args.out_dir)
    ctr = 0
    start_time = datetime.now()
    total_tracks_so_far = 0
    last_success = None
    stop_writing = False
    for subdir in os.listdir(args.data_dir):
        for subsubdir in os.listdir(os.path.join(args.data_dir, subdir)):
            for subsubsubdir in os.listdir(os.path.join(args.data_dir, subdir, subsubdir)):
                for track_name in os.listdir(os.path.join(args.data_dir, subdir, subsubdir, subsubsubdir)):
                    track_id = track_name[:track_name.rfind('.')]
                    full_filename = os.path.join(args.data_dir, subdir, subsubdir, subsubsubdir, track_name)
                    if full_filename == args.stop_at:
                        stop_writing = True
                    if start:
                        if track_id not in mappings.keys():
                            progress_file.write("No MSD -> musiXmatch mapping for %s\n" % full_filename)
                            # print "No MSD -> musiXmatch mapping for %s" % full_filename
                        else:
                            # print "Getting lyrics for %s and all similar songs" % full_filename
                            progress_file.write("Getting lyrics for %s and all similar songs\n" % full_filename)

                            similars = get_similar(json.load(codecs.open(full_filename, 'r', encoding='utf-8'), encoding='utf-8'))
                            similars.insert(0, track_id)

                            for song_id in similars:
                                if song_id not in mappings.keys():
                                    progress_file.write("\tNo MSD -> musiXmatch mapping for %s\n" % song_id)
                                    # print "\tNo MSD -> musiXmatch mapping for %s" % song_id
                                elif song_id in loaded_files:
                                    progress_file.write("\tAlready loaded lyrics for %s\n" % song_id)
                                    # print "\tAlready loaded lyrics for %s" % song_id
                                else:
                                    while True:
                                        if curr_api_key < len(API_KEYS):
                                            mxm_id = mappings[song_id]
                                            status, lyrics = get_lyrics(mxm_id, API_KEYS[curr_api_key])
                                            if status == 200:
                                                loaded_files.append(song_id)
                                                if song_id == track_id:
                                                    total_tracks_so_far += 1
                                                    ctr +=1
                                                    if ctr % 50 == 0:
                                                        print "Finished getting lyrics for %d tracks" % ctr
                                                    if ctr == args.lim:
                                                        stop_writing = True
                                                progress_file.write("\tSuccessfully got lyrics for %s\n" % song_id)
                                                # print "\tSuccessfully got lyrics for %s" % song_id
                                                last_success = full_filename
                                                outfile = codecs.open(args.out_dir+'/'+song_id+'.txt', 'w', encoding='utf-8')
                                                outfile.write(lyrics)
                                                outfile.close()
                                                break
                                            elif status == 400 or status == 404 or status == 403:
                                                progress_file.write("\tNo resource found for %s\n" % song_id)
                                                # print "\tNo resource found for %s" % song_id
                                                break
                                            elif status == 401 or status == 402:
                                                curr_api_key += 1
                                                progress_file.write("\tSwitching to API key #%d\n" % (curr_api_key+1))
                                                print "\tSwitching to API key #%d" % (curr_api_key+1)
                                            elif status == NOT_ENGLISH:
                                                progress_file.write("\tNot English: %s\n" % song_id)
                                                # print "\tNot English: %s" % song_id
                                                break
                                        else:
                                            progress_file.write("Used up all API keys for today!\n")
                                            print "Used up all API keys for today!\n"
                                            stop_writing = True
                                            break

                    else:
                        if track_id in loaded_files:
                            total_tracks_so_far += 1
                        if full_filename == args.start_from:
                            print "Total tracks so far: %d" % total_tracks_so_far
                            progress_file.write("Starting writing from file after %s\n" % full_filename)
                            print "Starting writing from file after %s" % full_filename
                            start = True

                    if stop_writing:
                        break
                if stop_writing:
                    break
            if stop_writing:
                break
        if stop_writing:
            break

    progress_file.write("Last file successfully written (use this as the --start_from argument for next run): %s\n" % last_success)
    print "Last file successfully written (use this as the --start_from argument for next run): %s\n" % last_success
    if curr_api_key < len(API_KEYS):
        progress_file.write("Last API key used: #%d\n" % (curr_api_key+1))

    progress_file.close()

    end_time = datetime.now()
    print "Total time taken: %s " % str(end_time - start_time)
    print "Total tracks processed so far: %d"  % total_tracks_so_far

if __name__ == '__main__':
    main()