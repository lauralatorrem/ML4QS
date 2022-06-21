##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util import util
from pathlib import Path
import os
import pandas as pd


def to_named_label(label):
    if label == 0:
        return "Awake"
    elif label == 1:
        return "N1"
    elif label == 2:
        return "N2"
    elif label == 3:
        return "N3"
    elif label == 5:
        return "REM"
    else:
        return "Unknown"


def convert_labels(csvfile):
    df = pd.read_csv(csvfile, sep=' ', names=["timestamp", "labels"], index_col=False)
    time = df["timestamp"].to_list()
    labels = df["labels"].to_list()
    label = []
    start = []
    end = []
    length = len(time)

    for i, x in enumerate(time):
        if i == 0:
            start.append(x)
            label.append(to_named_label(labels[i]))
        else:
            if labels[i] != labels[i - 1]:
                start.append(x)
                label.append(to_named_label(labels[i]))

            if i < length - 1:
                if labels[i] != labels[i + 1]:
                    end.append(x)
            else:
                end.append(x)
    df2 = pd.DataFrame(label)
    df2.columns = ["label"]
    df2["label_start"] = start
    df2["label_end"] = end
    df2.to_csv(csvfile[:-3]+'csv', index=False)

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/physionet.org/cropped/46343')
USER_ID = DATASET_PATH.name
RESULT_PATH = Path('./intermediate_datafiles/')
RESULT_FNAME = 'physionet_example.csv'

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

for filename in os.listdir(DATASET_PATH):
    if 'psg.out' in filename:
        convert_labels(str(DATASET_PATH)+'/'+filename)


# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
milliseconds_per_instance = 250

print('Please wait, this will take a while to run!')

print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

# Create an initial dataset object with the base directory for our data and a granularity
dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

# Add the selected measurements to it.

# We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
# and aggregate the values per timestep by averaging the values
dataset.add_numerical_dataset(USER_ID+'_cleaned_motion.out', 'timestamps', ['x','y','z'], 'avg', 'acc_phone_')

# We add the heart rate (continuous numerical measurements) and aggregate by averaging again
dataset.add_numerical_dataset(USER_ID+'_cleaned_hr.out', 'timestamps', ['rate'], 'avg', 'hr_watch_')

# We add the labels provided by the users. These are categorical events that might overlap. We add them
# as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
# occurs within an interval).
dataset.add_event_dataset(USER_ID+'_cleaned_psg.csv', 'label_start', 'label_end', 'label', 'binary')

# Get the resulting pandas data table
dataset = dataset.data_table

# And print a summary of the dataset.
util.print_statistics(dataset)

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through

print('The code has run through successfully!')