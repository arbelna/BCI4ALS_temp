def moving_avg(arr, window_size):
    import numpy as np
    for j in range(arr.shape[0]):
        moving_averages = [arr[j, 0]]
        i = 0
        while i < arr.shape[1] - window_size + 1:
            window_average = np.sum(arr[j, i:i + window_size]) / window_size
                       # window in moving average list
            moving_averages.append(window_average)
            # Shift window to right by one position
            i += 1
        if j == 0:
            new_arr = np.zeros((arr.shape[0], len(moving_averages)-1))
        new_arr[j, :] = moving_averages[1:]
    return new_arr

