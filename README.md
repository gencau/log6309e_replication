# Replication and extension of RQ1 for paper "On the effectiveness of log representations for log-based anomaly detection.

Requires the Drain parser, available here: https://github.com/logpai/logparser/tree/main/logparser/Drain
* Prior to running the scripts, install the logparser package from  https://github.com/logpai/logparser: python setup.py install
* Requirements.txt contains required packages: pip install -r requirements.txt

  # Extention of paper
  * Using statistical ranking to rank models based on performance. Resampling is done through time-based cross-validation
  * Gini importance evaluation for Decision Tree, chosen for best interpretability
  * Correlation analysis done with PCA analysis and redundancy analysis (VIF). Performance is compared with original models
  * Perform random splitting and compare the results with the original, time-based split.
