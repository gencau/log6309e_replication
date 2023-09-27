from logparser.Drain import LogParser

output_dir = 'Thunder_result/'  # The output directory of parsing results
log_file   = "Thunderbird_10m.log"  # The input log file name
log_format = "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>"
regex      = [
             r'(\d+\.){3}\d+',
             r'((a|b|c|d)n(\d){2,}\ ?)+', # a|b|c|dn+number
             r'\d{14}(.)[0-9A-Z]{10,}@tbird-#\d+#', # message id
             r'(?![0-9]+\W)(?![a-zA-Z]+\W)(?<!_|\w)[0-9A-Za-z]{8,}(?!_)',      # char+numbers,
             r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # ip address
             r'\d{8,}',   # numbers + 8
             r'(?<=:)(\d+)(?= )',    # parameter after :
             r'(?<=pid=)(\d+)(?= )',   # pid=XXXXX
             r'(?<=Lustre: )(\d+)(?=:)', # Lustre:
             r'(?<=,)(\d+)(?=\))'
              ] # regex for Thunderbird dataset

st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = LogParser(log_format, indir="data", outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)