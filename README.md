# juravinski
2021-data analysis

Data:
pre-processed and labeled 3 axis accelerometer data for all 30 subjects. There are two data trials, contain 4 poses and 3 sensors.
0-'Lying', 1-'Sitting', 2-'Standing', 3-'Walking'
Data are splitted into 2 seconds sliding windows, ready to be used for model training.
SubXX_train_data.txt stores the X
SubXX_train_label.txt stores corresponding y

Code:
python files name with ptm_XXX stands for the pooling task models for human activity recognition files.
Separate models were trained for each sensor location or their combinations.

