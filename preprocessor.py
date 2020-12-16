import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

class Preprocessor:
    def __init__(self):
        pass
    
    def signal_creator(self, MIT_signal_whole):
        MIT_data = []
        for i in range(len(MIT_signal_whole[0])):
            MIT_data.append(MIT_signal_whole[0][i][0])
        MIT_data = np.array(MIT_data)
        nulls = np.isnan(MIT_data)
        l=[]
        for i in range(len(nulls)):
            if nulls[i] == True:
                MIT_data[i] = 0
                l.append(i) 
        MIT_data[l] = np.mean(MIT_data)
        return MIT_data

    def moving_average(self, l, N):
        sum = 0
        result = list( 0 for x in l) 
        for i in range( 0, N ):
            sum = sum + l[i]
            result[i] = sum / (i+1)
        for i in range( N, len(l) ):
            sum = sum - l[i-N] + l[i]
            result[i] = sum / N
        return result

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def starting_point(self, signal,t,fs):
        M = np.amax(signal[0:300])
        index_M = np.where(signal == M)
        signal_start = signal[index_M[0][0]:-1]
        time_M = np.arange(signal_start.size)/fs
        return signal_start,time_M

    def signal_preprocessor(self, data, time, uu, mm):
        mean = np.mean(data)
        data = data - mean

        data = self.moving_average(data,5)
        data = np.array(data)

        data = self.butter_highpass_filter(data, 1, uu, 5)

        MIT_signal = self.butter_lowpass_filter(data, 1, mm, 5)
        fs = 30
        time = np.arange(MIT_signal.size)/fs
        time = time/10

        return MIT_signal,time

    def ready_signal(self, directory,freq1):
        s = wfdb.io.rdsamp(directory)
        s = self.signal_creator(s)
        t = np.arange(s.size)/freq1
        s,t= self.signal_preprocessor(s, t, freq1, 30)
        s = wfdb.processing.normalize_bound(s, lb=-1, ub=1)
        return s,t

    def collect_signals(self, directories, freq1):
        pass

