from logparser.Drain import LogParser

output_dir = 'BGL_result/'  # The output directory of parsing results
log_file   = "BGL.log"  # The input log file name
log_format = "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>"
regex      = [
             r"core\.\d+",
             r'(?<=r)\d{1,2}',
             r'(?<=fpr)\d{1,2}',
             r'(0x)?[0-9a-fA-F]{8}',
             r'(?<=\.\.)0[xX][0-9a-fA-F]+',
             r'(?<=\.\.)\d+(?!x)',
             r'\d+(?=:)',
             r'^\d+$',  #only numbers
             r'(?<=\=)\d+(?!x)',
             r'(?<=\=)0[xX][0-9a-fA-F]+',
             r'(?<=\ )[A-Z][\+|\-](?= |$)',
             r'(?<=:\ )[A-Z](?= |$)',
             r'(?<=\ [A-Z]\ )[A-Z](?= |$)'
             ] # regex for BGL dataset

st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

# It was required to modify Drain, otherwise there was a charset error
# Added option 'encoding="utf8"' at line 331
parser = LogParser(log_format, indir="data", outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)