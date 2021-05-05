import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given mnist X and y data and a particular numeral, returns a numpy array
# containing all instances of that numeral in the dataset
def get_MNIST_Numeral(numeral, X_data, y_data):
    out = []
    count = 0
    for X, y, in zip(X_data, y_data):
        if y == numeral:
            out.append(X)
            count +=1
    print(f'saved {count} {numeral}s')
    return np.array(out) 

# Given a list of images (each a numpy array)
# plots each image
def show_images(ims):
    for image in ims:
        plt.imshow(image, cmap='gray')
        plt.show() 

# Given mnist X and y data and a series of numerals as a list,
# this function will create a numpy array of images of the numerals
# in the series repeated num_repeats times.
def make_repeating_series(X_data, y_data, series, num_repeats):
    out_X = []
    out_y = []
    series_count = 0
    count = 0
    item = series.pop(0)
    series.append(item)
    while series_count != num_repeats:
        #print('start cycle through data')
        for X, y in zip(X_data, y_data):
            if y == item:
                out_X.append(X)
                out_y.append(y)
                item = series.pop(0)
                series.append(item)
                count+=1
                if count==len(series):
                    series_count+=1
                    count=0
                if series_count == num_repeats:
                    break
    return np.array(out_X), np.array(out_y)

# Given an array of numerals values y and a list of the numerals in the array,
# Returns an array of one-hot encoded vectors, encoded by position in the list.
# Example: if List = [1,4,5] and the first element of y is a 5
# then a numpy array [0,0,1] would be added to the output array
def make_one_hot_y_for_series(y, s_list):
    out = []
    for value in y:
        hot = np.zeros(len(s_list))
        if value in s_list:
            position = s_list.index(value)
            hot[position]=1
            out.append(hot)
    return np.array(out)

# Convert MNIST y data to 0 or 1 array for logistic regression ono
# a single numeral
def make_y_for_logistic(y, numeral):
    out = []
    for value in y:
        if value == numeral:
            out.append(1)
        else:
            out.append(0)
    return np.array(out)

# Given MNIST X and y data and a specific numeral, this function
# Returns a new dataset out_x and out_y with the number of the specific
# numeral in the new dataset, say the numeral 5, is equal to the number
# of all other numerals in the dataset.
def make_balanced_set(X, y, numeral, num_points, seed=None):
    flag = False
    out_x = []
    out_y = []
    total = 0
    while total != num_points:
        print('start cycle through data')
        for Xx, yy in zip(X,y):
            if flag == True:
                if yy == numeral:
                    out_x.append(Xx)
                    out_y.append(yy)
                    flag = False
                    total+=1
                    if total == num_points:
                        break
            else:
                if yy != numeral:
                    out_x.append(Xx)
                    out_y.append(yy)
                    flag = True
                    total+=1
                    if total == num_points:
                        break
    print(f'saved {total} images')
    df = pd.DataFrame({'x':out_x, 'y':out_y})
    if seed != None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1).reset_index(drop=True)
    out_x = list(df.x.values)
    out_y = list(df.y.values)
    return np.array(out_x),  np.array(out_y)

# plotting accuracy and loss
def plot_loss_and_accuracy(history):
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('model loss vs Epoch', fontsize = 14)
    ax[0].set_ylabel('Loss', fontsize = 14)
    ax[0].set_xlabel('Epoch', fontsize = 14)
    ax[0].tick_params(labelsize = 14)
    ax[0].legend(['train', 'val'], loc='best', fontsize = 12)
    ax[0].grid(True, lw = 1.5, ls = '--', color='gray', alpha = 0.3)

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('model accuracy vs Epoch', fontsize = 14)
    ax[1].set_ylabel('Loss', fontsize = 14)
    ax[1].set_xlabel('Epoch', fontsize = 14)
    ax[1].tick_params(labelsize = 14)
    ax[1].legend(['train', 'val'], loc='best', fontsize = 12)
    ax[1].grid(True, lw = 1.5, ls = '--', color='gray', alpha = 0.3)
    plt.show()  