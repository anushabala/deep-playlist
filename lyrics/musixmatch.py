__author__ = 'anushabala'
from argparse import ArgumentParser


API_KEYS=['2fabf028b619a95f4f7be3e74c861605',
          '859a94723b830c48b6d8d3ec9560a2c5',
          '029b2743a7c0371b8076a8090f999af1']


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str, help='Location of lastFM dataset to get lyrics from')
    parser.add_argument('--save_progress', type=str, default='./lyrics_scraper.progress', help='File to log progress of scraper to')
    parser.add_argument('--lim', type=int, default=-1, help='Maximum number of songs to request lyrics for')


if __name__ == '__main__':
    main()