import numpy as np

class preprocessor:
    def __init__(self):
        pass

    def signal_creator(self, raw_signal):
        signal_data = []
        for i in range(len(raw_signal[0])):
            signal_data.append(raw_signal[0][i][0])
        signal_data = np.array(signal_data)
        return signal_data

    def clear_null_values(self, signal_data):
        nulls = np.isnan(signal_data)
        null_data_indexes = []
        for i in range(len(nulls)):
            if nulls[i] == True:
                null_data_indexes.append(i)
                signal_data[i] = 0
            signal_data[i] = np.mean(signal_data)

if __name__ == '__main__':
    prep = preprocessor()

    # prep.signal_creator()
