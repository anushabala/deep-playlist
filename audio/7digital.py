from collections import defaultdict
import json
import urllib
import sys

__author__ = 'anushabala'

from argparse import ArgumentParser
import os
import sqlite3
# import urllib
# import hmac
# from hashlib import sha1
import random
import oauth2 as oauth
import wave
import pickle

API_KEYS = [('7dwtf7bqynwd', 'bevdf4pbrhdu8jwa'),
            ('7dhwct63x49f', 'r4vd3xa2sau7aqpm'),
            ('7dbwrva6ugm8', 'tfn52qxve937nz4m'),
            ('7desbz4zxq3c', '86uq8fw5sq57jgzc'),
            ('7dvycbkpgwqh', 'jjnydsp2yjccrrzy')]

trackid_url = 'http://api.7digital.com/1.2'
trackid_endpoint = 'track/match/byId'
# preview_url = 'http://api.7digital.com/1.2'
preview_url = 'http://previews.7digital.com'
# preview_endpoint = 'track/preview'
preview_endpoint = 'clip'
no_sdl_id = 'NO_SDL_ID'
no_preview = 'NO_PREVIEW'
save_error = 'SAVE_ERR'
auth_error = 401
no_resource = 2001
internal_error = 9001


def get_7digital_trackid(db_cursor, msd_id):
    db_cursor.execute("SELECT track_7digitalid FROM songs WHERE track_id=?", (msd_id,))
    x = db_cursor.fetchone()
    if x:
        return x[0], False
    else:
        return None, True


def get_oauth_nonce(length=8):
    return ''.join([str(random.randint(0, 9)) for i in range(length)])


def get_access_token(key, secret):
    consumer = oauth.Consumer(key=key, secret=secret)
    request_url = "https://api.7digital.com/1.2/oauth/requesttoken"
    client = oauth.Client(consumer)
    resp, content = client.request(request_url, "POST")


def get_preview_url(api_key, api_secret, sdl_id):
    consumer = oauth.Consumer(api_key, api_secret)
    request_url = "http://previews.7digital.com/clip/%s" % sdl_id

    req = oauth.Request(method="GET", url=request_url, is_form_encoded=True)

    req['oauth_timestamp'] = oauth.Request.make_timestamp()
    req['oauth_nonce'] = oauth.Request.make_nonce()
    req['country'] = 'US'
    sig_method = oauth.SignatureMethod_HMAC_SHA1()

    req.sign_request(sig_method, consumer, token=None)

    url = req.to_url()
    return url, False


def save_as_audio(preview_url, dst_file):
    try:
        bytestream = urllib.urlopen(preview_url)

        bytes = bytestream.read(1024)
        if "Track not found" in bytes:
            return no_resource, 'track not found'
        out_file = open(dst_file, 'wb')
        # read 1024 bytes at a time

        while bytes:
            out_file.write(bytes)
            bytes = bytestream.read(1024)

        out_file.close()
        return 0, 'success'
    except IOError as error:
        error_code = error[1]
        msg = error
        if isinstance(error, tuple) and len(error) > 2:
            msg = error[2]
        # print error_code
        # print error[2]
        # print str(error[3])
        #
        return error_code, msg


def get_track(db_cursor, msd_id, dst_file, curr_api_key, curr_api_secret):
    sdl_id, err = get_7digital_trackid(db_cursor, msd_id)
    if err:
        return no_sdl_id, 'no 7digital id'

    preview_url, err = get_preview_url(curr_api_key, curr_api_secret, sdl_id)
    if err:
        return no_preview, 'no preview'

    err, msg = save_as_audio(preview_url, dst_file)
    return err, msg


def get_already_loaded(dirname):
    return [name.replace(".mp3", "") for name in os.listdir(dirname)]


def main():
    parser = ArgumentParser()
    parser.add_argument('--dir', default='../data/lyrics/train', help='Path to data to get audio for')
    parser.add_argument('--mappings_file', default='../lastfm_train_mappings.txt')
    parser.add_argument('--lim', type=int, default=-1, help='Maximum number of tracks to get audio from')
    parser.add_argument('--save_progress', default='./audio_scraper.progress')
    parser.add_argument('--metadata_db', default='../data/metadata/track_metadata.db', help='Path to SQLite database containing track metadata')
    parser.add_argument('--start_from', default='none', help='Track ID to start getting audio from.')
    parser.add_argument('--out_dir', default='../data/audio/train')
    parser.add_argument('--print_every', default=50, type=int)

    args = parser.parse_args()

    progress_file = open(args.save_progress, 'a')
    conn = sqlite3.connect(args.metadata_db)
    cursor = conn.cursor()
    ctr = 0
    curr_api_key = 0
    last_written = None

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # get_access_token(API_KEYS[0], "bevdf4pbrhdu8jwa10734")
    already_loaded = get_already_loaded(args.out_dir)
    stop_writing = False
    start_writing = True if args.start_from == 'none' else False
    loaded_so_far = len(already_loaded)
    print "# of tracks already loaded: %d" % loaded_so_far
    mapping_file = open(args.mappings_file, 'r')
    examples_per_id = defaultdict(int)

    num_pairs_loaded = 0
    if os.path.exists('audio_error_ids.pkl'):
        error_ids = pickle.load(open('audio_error_ids.pkl','rb'))
    else:
        error_ids = set()
    for line in mapping_file.readlines():

        cols = line.strip().split(',')
        ids = [cols[0], cols[1]]
        if cols[0] in already_loaded and cols[1] in already_loaded:
            num_pairs_loaded += 1
            continue
        if cols[0] in error_ids or cols[1] in error_ids:
            print "Skipping pair: %s" % line
            continue
        if stop_writing:
            break
        if start_writing:
            if examples_per_id[cols[0]] >= 2:
                # print "Got two pairs for %s" % cols[0]
                continue
            success = 0
            for msd_id in ids:
                if stop_writing:
                    break
                out_file = os.path.join(args.out_dir, msd_id) + '.mp3'
                if msd_id in already_loaded:
                    progress_file.write("Already wrote audio for %s\n" % msd_id)
                    success += 1
                    print "Already wrote audio for %s" % msd_id
                    continue
                while True:
                    err, msg = get_track(cursor, msd_id, out_file, API_KEYS[curr_api_key][0], API_KEYS[curr_api_key][1])
                    if err == auth_error:
                        progress_file.write('Authentication error while getting audio for %s\n' % msd_id)
                        curr_api_key += 1
                        if curr_api_key < len(API_KEYS):
                            progress_file.write('Trying next API key (#%d)\n' % (curr_api_key+1))
                        else:
                            progress_file.write('Used up all API keys for today!\n')
                            print 'Used up all API keys for today!'
                            stop_writing = True
                            break
                    elif err == no_resource:
                        progress_file.write('No resource found for %s\n' % msd_id)
                        error_ids.add(msd_id)
                        break
                    elif err == no_preview:
                        progress_file.write('No preview found for %s\n' % msd_id)
                        error_ids.add(msd_id)
                        break
                    elif err == no_sdl_id:
                        progress_file.write('No 7digital ID for %s\n' % msd_id)
                        error_ids.add(msd_id)
                        break
                    elif err == 0:
                        success += 1
                        progress_file.write('Wrote audio for track %s to %s\n' % (msd_id, out_file))
                        break
            if success == 2:
                ctr += 1
                examples_per_id[cols[0]] += 1
                last_written = line
                if ctr % args.print_every == 0:
                    progress_file.write("Got audio for %d pairs\n" % ctr)
                    print "Got audio for %d pairs" % ctr
                if ctr >= args.lim:
                    stop_writing = True
                    break

        else:
            if cols[0] == args.start_from:
                print 'Started writing audio from %s' % cols[0]
                start_writing = True

    progress_file.write('Last line written: %s\n' % last_written)
    loaded_so_far += ctr
    progress_file.write("Total tracks loaded so far: %d\n" % loaded_so_far)
    print "Total tracks loaded so far: %d" % loaded_so_far
    print "Total number of pairs loaded so far: %d" % (num_pairs_loaded + ctr)

    pickle.dump(error_ids, open('audio_error_ids.pkl', 'wb'))

if __name__ == "__main__":
    main()