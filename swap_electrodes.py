""" EEG Electrodes list/map: we have data from 17 electrodes
0 = Pz, on center line
1 = P3, mirror of 6, P4
2 = P7, mirror of 7, P8
3 = O1, mirror of 5, O2
4 = Oz, on center line
5 = O2, mirror of 3, O1
6 = P4, mirror of 1, P3
7 = P8, mirror of 2, P7
8 = P1, mirror of 16, P2
9 = P5, mirror of 15, P6
10 = PO7, mirror of 14, PO8
11 = PO3, mirror of 13, PO4
12 = POz, on center line
13 = PO4, mirror of 11, PO3
14 = PO8, mirror of 10, PO7
15 = P6, mirror of 9, P5
16 = P2, mirror of 8, P1

So, for example, to swap O1 with O2, you would use:
swap_electrodes([0, 1, 2, 5, 4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

BEFORE RUNNING THIS CODE:
To rename data folder, use:
mv data data_original

To make new data folder, use:
mkdir data
mkdir data/thingseeg2_preproc
mkdir data/thingseeg2_preproc/sub-01

Then you can run this code, the folders will exist in the right places
If you want to change subjects or use multiple subjects, just replace sub-01
"""
import numpy as np
import argparse

def swap_electrodes(new_order: list[int], subject: int):
    """Swap electrodes to match the new ordering of the electrodes.
    
    Move the original data from data/ to data_original/
    Will do only one subject.
    """
    # Make sure the original data is in folder data_original
    # Only looking at subject #1, change *both* of the next two lines if you 
    #    want a different subject
    path_orig = f'data_original/thingseeg2_preproc/sub-{subject:0>2}/'
    path_new = f'data/thingseeg2_preproc/sub-{subject:0>2}/'
    # First, swap the training data
    filename = path_orig + 'preprocessed_eeg_training.npy'
    with open(filename, 'rb') as fin:
        data = np.lib.format.read_array(
            fin, 
            allow_pickle=True
        )
    # Extract the dictionary from the 0-dim ndarray
    data_dict = data.item()
    print("EEG Channels: ", data_dict['ch_names'])
    # Get the actual EEG readings
    data_eeg = data_dict['preprocessed_eeg_data']
    # Print the first value for *each* electrode you want to swap
    print("Sample Electrode values before switch (Training):")
    for i, entry in new_order:
        if entry != i + 1:
            print(f"{i}: {data_eeg[0][0][i][0]:.4f}", end="\t")
    print()
    # Move the electrodes to the first dimension to make it easy to swap
    data_eeg = np.transpose(data_eeg, [2, 1, 0, 3])
    # This is where you swap electrodes
    data_eeg = data_eeg[new_order]
    # Move the electrodes back so the shape of the data is where it was
    data_eeg = np.transpose(data_eeg, [2, 1, 0, 3])
    # Re-assign the EEG readings in the data object to the newly swapped data
    data_dict['preprocessed_eeg_data'] = data_eeg
    # Print out the first value for each electrode you swapped to verify
    print("Swapped electrode values (Training):")
    for i, entry in new_order:
        if entry != i + 1:
            print(f"{i}: {data_eeg[0][0][i][0]:.4f}", end="\t")
    print()
    filename = path_new + 'preprocessed_eeg_training.npy'
    # np.save(filename, data, allow_pickle=True)

    # Next, do it again with the test data
    filename = path_orig + 'preprocessed_eeg_test.npy'
    with open(filename, 'rb') as fin:
        data = np.lib.format.read_array(
            fin, 
            allow_pickle=True
        )
    # Extract the dictionary from the 0-dim ndarray
    data_dict = data.item()
    # Get the actual EEG readings
    data_eeg = data_dict['preprocessed_eeg_data']
    print(data_eeg.shape)
    # Print the first value for each electrode you want to swap
    print("Sample Electrode values before switch (Test):")
    for i, entry in new_order:
        if entry != i + 1:
            print(f"{i}: {data_eeg[0][0][i][0]:.4f}", end="\t")
    print()
    # Move the electrodes to the first dimension to make it easy to swap
    data_eeg = np.transpose(data_eeg, [2, 1, 0, 3])
    # This is where you swap electrodes
    data_eeg = data_eeg[new_order]
    # Move the electrodes back so the shape of the data is where it was
    data_eeg = np.transpose(data_eeg, [2, 1, 0, 3])
    # Re-assign the EEG readings in the data object to the newly swapped data
    data_dict['preprocessed_eeg_data'] = data_eeg
    # Print out the first value for each electrode you swapped to verify
    print("Swapped electrode values (Test):")
    for i, entry in new_order:
        if entry != i + 1:
            print(f"{i}: {data_eeg[0][0][i][0]:.4f}", end="\t")
    print()
    filename = path_new + 'preprocessed_eeg_test.npy'
    # np.save(filename, data, allow_pickle=True)


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--subject", default=1, type=int,
        help="Number of subject to work on, right now 1-10")
ap.add_argument("-e", "--electrodes", required=True,
        help="List of electrodes to swap, in Python list format, in quotes.\n" + 
        "You only need to enter one half of a matched pair.\n" + 
        "So, for ex, if you enter '[1,2]' this will swap 1(P3) with 6(P4)\n" +
        "and 2(P7) with 7(P8).\n")
args = ap.parse_args()
electrodes = eval(args.electrodes)
pairs = ((1,6), (2,7), (3,5), (8,16), (9,15), (10,14), (11,13))
out_of_place = []
for e in electrodes:
    found = False
    for pair in pairs:
        if e in pair:
            out_of_place.extend(pair)
            found = True
    if not found:
        print(f"Electrode {e} not swappable, skipping...")
out_of_place = list(set(out_of_place))
if not out_of_place:
    print("No electrodes given can be swapped, quitting.")
    quit()
new_order = []
for i in range(17):
    if i not in out_of_place:
        new_order.append(i)
    else:
        for pair in pairs:
            if i == pair[0]:
                new_order.append(pair[1])
                break
            elif i == pair[1]:
                new_order.append(pair[0])
                break
swap_electrodes(new_order, args.subject)
