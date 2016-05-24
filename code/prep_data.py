__author__ = 'kalpit'

import glob
import fnmatch
import os
import json


def get_available_tracks(base):
    file_locs = glob.glob(base + '*')
    file_names = [x[x.rfind('/') + 1:-4] for x in file_locs]
    return file_locs, file_names


def write_similar_available_tracks(base, a_file_names, fname):
    fw = open(fname, 'w')
    # Pairs = [] # list of lists. [track_id_1, track_id_2, similarity_score]

    c = 0
    d = 0
    e = 0
    for root, dirnames, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, '*.json'):
            d += 1
            if d % 10000 == 0:
                print d
            if filename[:-5] not in a_file_names:  # removing extension
                continue
            e += 1
            # Parse names of similar tracks
            with open(root + '/' + filename, 'r') as f:
                data = json.load(f)
                data = data['similars']
                c += len(data)
                # tracks = [[filename[:-5], x[0], x[1]] for x in data if x[0] in a_file_names]
                for x in data:
                    if x[0] in a_file_names:
                        fw.write(filename[:-5] + ',' + x[0] + ',' + str(x[1]) + '\n')
                        # Pairs.extend(tracks)
    print c
    print 'files in ' + base + '               : ' + str(d)
    print 'files in ' + base + ' and available : ' + str(e)
    fw.close()
    return


if __name__ == '__main__':
    ##### Available Tracks #####
    BaseDir_Available_Tracks = '../lyrics/data/lyrics/train/'
    a_file_locs, a_file_names = get_available_tracks(BaseDir_Available_Tracks)
    print 'available files : ', len(a_file_names)

    ##### Similar (and available) Tracks #####
    BaseDir_Similar_Tracks = '../lastfm_train'
    similar_tracks_fname = '../lastfm_train_mappings.txt'
    write_similar_available_tracks(BaseDir_Similar_Tracks, a_file_names, similar_tracks_fname)
    # print len(Pairs)
