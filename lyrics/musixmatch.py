__author__ = 'anushabala'
from argparse import ArgumentParser
import os
import urllib
import json
import codecs


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

    ctr = 0
    last_success = None
    stop_writing = False
    for subdir in os.listdir(args.data_dir):
        for subsubdir in os.listdir(os.path.join(args.data_dir, subdir)):
            for subsubsubdir in os.listdir(os.path.join(args.data_dir, subdir, subsubdir)):
                for track_name in os.listdir(os.path.join(args.data_dir, subdir, subsubdir, subsubsubdir)):
                    id = track_name[:track_name.rfind('.')]
                    full_filename = os.path.join(args.data_dir, subdir, subsubdir, subsubsubdir, track_name)
                    if start:
                        if id not in mappings.keys():
                            progress_file.write("No MSD -> musiXmatch mapping for %s\n" % full_filename)
                        else:
                            while True:
                                if curr_api_key < len(API_KEYS):
                                    mxm_id = mappings[id]
                                    status, lyrics = get_lyrics(mxm_id, API_KEYS[curr_api_key])
                                    if status == 200:
                                        ctr +=1
                                        progress_file.write("Successfully got lyrics for %s\n" % full_filename)
                                        last_success = full_filename
                                        outfile = codecs.open(args.out_dir+'/'+id+'.txt', 'w', encoding='utf-8')
                                        outfile.write(lyrics)
                                        outfile.close()

                                        if ctr == args.lim:
                                            stop_writing = True

                                        if ctr % 50 == 0:
                                            print "Finished getting lyrics for %d tracks" % ctr
                                        break
                                    elif status == 400 or status == 404 or status == 403:
                                        progress_file.write("No resource found for %s\n" % full_filename)
                                        break
                                    elif status == 401 or status == 402:
                                        curr_api_key += 1
                                        progress_file.write("Switching to API key #%d\n" % (curr_api_key+1))
                                    elif status == NOT_ENGLISH:
                                        progress_file.write("Not English: %s" % full_filename)
                                        break
                                else:
                                    progress_file.write("Used up all API keys for today!\n")
                                    stop_writing = True
                                    break
                    else:
                        if full_filename == args.start_from:
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
    if curr_api_key < len(API_KEYS):
        progress_file.write("Last API key used: #%d\n" % (curr_api_key+1))

    progress_file.close()

if __name__ == '__main__':
    main()