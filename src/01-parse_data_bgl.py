#!/usr/bin/env python

import sys
from logparser.Drain import LogParser


def structure_bgl(input_dir, output_dir, log_file):
    log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL log format
    # Regular expression list for optional preprocessing (default: [])
    regex = [r'core\.\d+',
             r'(?<=r)\d{1,2}',
             r'(?<=fpr)\d{1,2}',
             r'(0x)?[0-9a-fA-F]{8}',
             r'(?<=\.\.)0[xX][0-9a-fA-F]+',
             r'(?<=\.\.)\d+(?!x)',
             r'\d+(?=:)',
             r'^\d+$',  # only numbers
             r'(?<=\=)\d+(?!x)',
             r'(?<=\=)0[xX][0-9a-fA-F]+',
             r'(?<=\ )[A-Z][\+|\-](?= |$)',
             r'(?<=:\ )[A-Z](?= |$)',
             r'(?<=\ [A-Z]\ )[A-Z](?= |$)'
             ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    parser.parse(log_file)


structure_bgl('../data/BGL/', '../data/BGL', 'BGL.log')
