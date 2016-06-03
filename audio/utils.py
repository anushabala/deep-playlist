import os
import json
__author__ = 'anushabala'


def create_mappings_file(dirname, out_file):
    out_file = open(out_file, 'w')
    ctr = 0
    for first in os.listdir(dirname):
        first_path = os.path.join(dirname, first)
        for second in os.listdir(first_path):
            second_path = os.path.join(first_path, second)
            for third in os.listdir(second_path):
                third_path = os.path.join(second_path, third)
                for name in os.listdir(third_path):
                    ctr += 1
                    id = name.replace(".json","")
                    f = os.path.join(third_path, name)
                    data = json.load(open(f, 'r'))
                    for similar in data["similars"]:
                        similar_id = similar[0]
                        score = similar[1]
                        out_file.write("%s,%s,%2.4f\n" % (id, similar_id, score))
                    if ctr % 100 == 0:
                        print "Finished %d files" % ctr
    out_file.close()


if __name__ == "__main__":
    create_mappings_file('../../lastfm/lastfm_test', '../lastfm_test_mappings.txt')