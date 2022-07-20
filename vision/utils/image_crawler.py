#!/usr/bin/python

import argparse
import os

from icrawler.builtin import GoogleImageCrawler

from utils import util


def crawl_and_download(download_dir, keyword, filter_type, filename):
    try:
        filters = {'type': filter_type}
        google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                            storage={'root_dir': download_dir})
        google_crawler.crawl(keyword=keyword, max_num=1000,
                             min_size=(200, 200), max_size=(1024, 1024), filters=filters)
    finally:
        util.write_files_to_file(download_dir, filename)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download_dir", type=str, default="./downloads/", help="download directory")
    parser.add_argument("-k", "--keyword", type=str, default="face", help="search keyword")
    parser.add_argument("-f", "--filename", type=str, default=None, help="file to save the list of downloaded files")
    parser.add_argument("-t", "--type", type=str, default='photo', help="Photo filter type [photo, face, clipart, "
                                                                        "linedrawing, animated]")
    args = parser.parse_args(args)
    if args.filename is None:
        args.filename = os.path.join(args.download_dir, "files.txt")
    return args


def demo():
    args = "-d download/ -k 'retail+products' -t photo -f download/files.txt".split()
    args = parse_args(args)
    print("Downloading '{}' of type '{}' into the folder '{}'. Final list will be written to the file '{}'".format(
        args.keyword, args.type, args.download_dir, args.filename))
    crawl_and_download(args.download_dir, args.keyword, args.type, args.filename)


def retail_items():
    args = "-d data/retail_test_images/ -k 'retail+store+with+pampers+whispers+gillette+oreo+shampoos+and" \
           "+chyawanprash' -t photo -f data/retail_test_images.txt".split()
    args = parse_args(args)
    print("Downloading '{}' of type '{}' into the folder '{}'. Final list will be written to the file '{}'".format(
        args.keyword, args.type, args.download_dir, args.filename))
    crawl_and_download(args.download_dir, args.keyword, args.type, args.filename)


if __name__ == "__main__":
    # args = parse_args()
    # print("Downloading '{}' of type '{}' into the folder '{}'. Final list will be written to the file '{}'".format(
    #     args.keyword, args.type, args.download_dir, args.filename))
    # crawl_and_download(args.download_dir, args.keyword, args.type, args.filename)

    retail_items()
