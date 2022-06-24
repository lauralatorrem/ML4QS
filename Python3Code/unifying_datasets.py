import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse
import os

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

DATA_PATH = Path('./intermediate_datafiles/physionet_patients')
DATASET_FNAME = 'chapter4_result.csv'

result_files = [x for x in os.listdir(DATA_PATH) if DATASET_FNAME in x]

datasets = []
pattern_columns = []
for file in result_files:
    df = pd.read_csv(DATA_PATH / file, index_col=0)
    pattern_columns.append([x for x in df.columns if 'temp_pattern' in x])
    datasets.append(df)

set_patterns = set(sum(pattern_columns, []))

for df, pcolumns, filename in zip(datasets, pattern_columns, result_files):
    new_columns = set_patterns - set(pcolumns)
    for col in new_columns:
        df[col] = 0
    outname = filename[:-10] + 'unified.csv'
    df.to_csv(DATA_PATH / outname)
