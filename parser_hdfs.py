#!/usr/bin/env python

from logparser.Drain import LogParser

output_dir = 'HDFS_result/'  # The output directory of parsing results
log_file   = "HDFS.log"  # The input log file name
log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
regex      = [
                r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"
              ] # regex for hdfs dataset

st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = LogParser(log_format, indir="data", outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)