#!/usr/bin/env python

from logparser.Drain import LogParser


def parse_to_csv(input_dir, output_dir, log_file):
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    # Regular expression list for optional preprocessing (default: [])
    regex = [
        r'blk_(|-)[0-9]+',  # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
    ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    parser.parse(log_file)
