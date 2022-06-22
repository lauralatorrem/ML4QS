##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'physionet_1360686_chapter3_result_final.csv'
RESULT_FNAME = 'physionet_1360686_chapter4_result.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))



def main():
    print_flags()
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]

         #please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.     
        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        for ws in window_sizes:

            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
            print("mean done")
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
            print("std done")
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'max')
            print("max done")
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'median')
            print("median done")
            dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'min')
            print("min done")
            #dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'slope')
            #print("slope done")

        for col in selected_predictor_cols:

            DataViz.plot_dataset(dataset, [col, col + '_temp_mean', col + '_temp_std', 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
            print("--- %s seconds ---" % (time.time() - start_time))

            DataViz.plot_dataset(dataset, [col, col + '_temp_max', col + '_temp_median', 'label'],
                                ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
            print("--- %s seconds ---" % (time.time() - start_time))

            DataViz.plot_dataset(dataset, [col, col + '_temp_min', 'label'],
                                ['exact', 'like', 'like'], ['line', 'line', 'points'])
            print("--- %s seconds ---" % (time.time() - start_time))

  
    if FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window size.
       
        fs = float(1000)/milliseconds_per_instance
        ws = int(float(10000)/milliseconds_per_instance)
        periodic_predictor_cols = ['acc_phone_x'
            , 'acc_phone_y', 'acc_phone_z','hr_watch_rate']
        dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, ws, fs)
        # Spectral analysis.
        DataViz.plot_dataset(dataset, ['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line','line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
        DataViz.plot_dataset(dataset, ['acc_phone_y_max_freq', 'acc_phone_y_freq_weighted', 'acc_phone_y_pse', 'label'],
                             ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
        DataViz.plot_dataset(dataset, ['acc_phone_z_max_freq', 'acc_phone_z_freq_weighted', 'acc_phone_z_pse', 'label'],
                             ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
        DataViz.plot_dataset(dataset, ['hr_watch_rate_max_freq', 'hr_watch_rate_freq_weighted', 'hr_watch_rate_pse', 'label'],
                             ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
    if FLAGS.mode == 'final':
        

        ws = int(float(0.5*60000)/milliseconds_per_instance)
        fs = float(1000)/milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
        # TODO: Add your own aggregation methods here
        dataset = NumAbs.abstract_numerical(dataset, ["hr_watch_rate"], ws, 'max')
        dataset = NumAbs.abstract_numerical(dataset, ["hr_watch_rate"], ws, 'min')


        DataViz.plot_dataset(dataset, ['acc_phone_x', 'pca_1', 'label'], ['like', 'like', 'like'], ['line', 'line', 'points'])

     
        CatAbs = CategoricalAbstraction()
        
        dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.02, int(float(5*60000)/milliseconds_per_instance), 3)


        periodic_predictor_cols = ['acc_phone_x'
                                    ,'acc_phone_y','acc_phone_z',]


        
        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)


        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]


        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset, ['acc_phone_x', 'pca_1', 'label'], ['like', 'like','like'], [  'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final']) 

    

    FLAGS, unparsed = parser.parse_known_args()
    
    main()