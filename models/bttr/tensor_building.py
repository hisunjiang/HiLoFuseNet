import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.io import loadmat as sploadmat
from scipy.io import matlab
from scipy.io import savemat as spsavemat
from scipy.signal import butter, filtfilt, hilbert
from scipy.sparse.linalg import spsolve
from scipy.stats import pearsonr, wilcoxon, zscore
from tensorly.datasets.synthetic import gen_image

from .bttr import BTTR


class DataDoesNotExist(Exception):
    """
    Used when a data loader is used within a config file that isn't registered to be used.
    """
    pass

class TensorData:
    r"""
        Helper Class to build a Tensor from a time series
    """

    @staticmethod
    def loadMat(filename : str):
        r"""
        This function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects

        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries

        Args:
            filename (str): Filename to be loaded (.mat file)
        """
        data = sploadmat(filename, struct_as_record=False)
        for key in data:
            if isinstance(data[key], matlab.mio5_params.mat_struct):
                data[key] = TensorData._todict(data[key])
        return data

    @staticmethod
    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries

        Args:
            matobj (matlab.mio5_params.mat_struct): convert a matlab object to a Python dictionary
        """
        dictionary = {}
        #noinspection PyProtectedMember
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                dictionary[strg] = TensorData._todict(elem)
            else:
                dictionary[strg] = elem
        return dictionary

    @staticmethod
    def saveMat(data, filename : str):
        r"""
        This function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects

        Args:
            filename (str): Filename to be loaded (.mat file)
        """
        return spsavemat(filename, data)

    @staticmethod
    def saveTensor(data, filename):
        r"""
        Save a Tensor to a python pickle file

        Args: 
            data (): Data to be saved to a pickle file
            filename (str): Pickle file containing the tensors
        """
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadTensor(filename):
        r"""
        Load a Tensor from a python pickle file

        Args: 
            filename (str): Pickle file containing the tensors
        """
        try:
            with open(str(filename) + '.pickle', "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError:
            # normal, somewhat expected
            raise DataDoesNotExist()
        except (AttributeError,  EOFError, ImportError, IndexError):
            # secondary errors
            # print(traceback.format_exc(e))
            raise DataDoesNotExist()
        except Exception:
            # everything else, possibly fatal
            # print(traceback.format_exc(e))
            raise DataDoesNotExist()

class TensorAnalysis:

    @staticmethod
    def statisticalSignificance(x, y):
        r"""
        Compute a Wilcoxon signed-rank test

        Args:
            x (np.ndarray): First set of observations
            y (np.ndarray): Second set of observations
        """
        return wilcoxon(x, y)

    @staticmethod
    def normalize(data):
        r"""
        Normalize data between 0 and 1

        Args:
            data (np.ndarray): Data to be normalized
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def pearson_correlation(truth, predicted, axis=0):
        r"""
        Calculate Pearson's correlation coefficients

        Args:
            truth (np.ndarray): The tensor containing the ground truth labels
            predicted (np.ndarray): The tensor containing the predicted labels. Is required to be the same shape as parameter ``truth``
            axis (int): axis to predict the Pearson's correlation along. Cannot be larger than the dimension of truth/predicted. If this is None, then the overall Pearson's correlation is given

        """
        if truth.shape != predicted.shape:
            raise Exception("Size of truth tensor shape " + str(truth.shape) + " does not match the predicted tensor shape " + str(predicted.shape))
        if len(truth.shape) <= axis:
            raise Exception("Axis " + str(axis) + " is too large (maximum size: " + str(len(truth.shape)) + ")")
        if axis is not None:
            truth = np.rollaxis(truth, axis)
            predicted = np.rollaxis(predicted, axis)
            return [pearsonr(truth[i], predicted[i]) for i in range(0, truth.shape[0])]
        else:
            return pearsonr(truth[:,0], predicted)

    @staticmethod
    def trainBTTR(X, Y, X_test, Y_test, nFactor, score_vector_matrix=False, plot=False):
        start = time.time()
        bttr = BTTR()
        bttr.train(X, Y, nFactor=nFactor, score_vector_matrix=score_vector_matrix)
        end = time.time()
        timing = end - start
        out = bttr.predict(X_test)
        corr = [TensorAnalysis.pearson_correlation(Y_test, pred, axis=1) for pred in out]

        if plot: TensorAnalysis.plotFingers(Y_test, out[TensorAnalysis.findMaxCorr(corr)])
        for c in corr:
            print([round(i[0], 4) for i in c])  # type: ignore
        modelX, modelY = bttr.getModel()
        plt.figure(figsize=(20,20))
        plt.imshow(modelX, aspect='auto', interpolation='none', cmap=mpl.colormaps["PiYG"])
        plt.colorbar()
        plt.figure(figsize=(20,20))
        plt.imshow(modelY, aspect='auto', interpolation='none', cmap=mpl.colormaps["PiYG"])
        plt.colorbar()
        plt.show()
        return corr, out, timing

    @staticmethod
    def findMaxCorr(correlations):
        max_idx = 0
        max_elem = None
        for i in range(0, len(correlations)):
            elem = sum(i for i, j in correlations[i])
            if max_elem is None or elem > max_elem:
                max_elem = elem
                max_idx = i
        return max_idx

    @staticmethod
    def plotFingers(truth, predicted):
        plt.plot(TensorAnalysis.normalize(predicted), label='Predicted')
        plt.plot(TensorAnalysis.normalize(truth), label='True')
        plt.legend()
        plt.show()



class TensorBuilder:
    r"""
        Helper Class to build a Tensor from a time series
    """

    @staticmethod
    def commonAverageReference(data):
        r"""
        Apply Common Average Referencing (CAR). The average of all signals is taken as reference and subtracted from all signals

        Args:
            data (np.ndarray): data to apply the CAR to.
        """
        return data - np.average(data)

    @staticmethod
    def bandpassFilter(data, frequencyBands, samplingRate):
        r"""
        Apply a 4th-order Butterworth band-pass filter along the specified frequency bands to extract the corresponding spectral amplitudes.

        Args:
            data (np.ndarray): The tensor with dimensions (time x channels)
            frequencyBands ((float, float) list): list of tuples of frequency bands
            samplingRate (int): sampling rate of the data
        """
        out = np.zeros((data.shape[0], data.shape[1], len(frequencyBands)))
        for i in range(0, len(frequencyBands)):
            filt = butter(N=4, Wn=frequencyBands[i], btype='bandpass', fs=samplingRate)
            signal = filtfilt(filt[0], filt[1], data)
            # signal = np.abs(hilbert(signal))
            out[:,:,i] = signal
        return out
    
    @staticmethod
    def zscore(data, axis=None):
        r"""
        Simple z-scoring function along the specified axis

        Args:
            data (np.ndarray): The tensor to be zscored
            axis (int): axis to zscore along. If this is None, then the overall zscore is given
        """
        return zscore(data, axis)  # type: ignore

    @staticmethod
    def find_first(item, vec):
        r"""
        return the index of the first occurence of item in vec
        
        Args:
            item: item to find in the vector
            vec (np.array): data to search
        """
        for i in range(len(vec)):
            if item == vec[i]:
                return i
        return -1

    @staticmethod
    def frequencyBins(data, frequencyBands, samplingRate, downSampleTo, preActionTime):
        res = []
        for i in range(samplingRate*preActionTime, data.shape[0]):
            res_item = np.zeros((data.shape[1], downSampleTo, len(frequencyBands)))
            for i_freq in range(0, len(frequencyBands)):
                tmp = data[i - samplingRate*preActionTime:i]
                filt = butter(N=4, Wn=frequencyBands[i_freq], btype='bandpass', fs=samplingRate)
                signal = filtfilt(filt[0], filt[1], tmp)
                signal = TensorBuilder.downSample(signal, samplingRate, downSampleTo)
                res_item[:,:,i_freq] = signal.T
            res_item = np.transpose(res_item, (0,2,1))
            res.append(res_item)
            if i%1000 == 0: print(i)
        return np.array(res)

    @staticmethod
    def selectTrials(data, target, samplingRate, preActionTime, postActionTime):
        r"""
        Select trials from data using dataglove activity (when markers aren't available)

        Args:
            data (np.ndarray): Data from which to select the trials
            target (np.array): 1-D Array to select trials
            samplingRate (int): sampling rate of the data
            preActionTime (int): length of time to select before the trial starts in seconds
            postActionTime (int): length of the trial in seconds
        """
        epochMask = np.zeros((target.shape[0]))
        epochMask[target[:, 0] > 1] = 1

        i = 0
        start = 0
        while i < data.shape[0]:
            i = start + TensorBuilder.find_first(1, epochMask[start:])
            if i < 0: break
            epochMask[i-int(preActionTime*samplingRate) : i+int(postActionTime*samplingRate)] = 1
            i = i+int(postActionTime*samplingRate)
            start = i+2*samplingRate
        
        target = target[epochMask > 0, :]
        data = data[epochMask > 0, :]

        return data, target

    @staticmethod
    def downSample(epoch, currentSampleRate, newSampleRate):
        r"""
            Downsample to another sample rate

            Args:
                epoch (np.ndarray): Timeseries to downsample
                currentSampleRate (int): The current sample rate of the data
                newSampleRate (int): The new (lower) sample rate of the data
        """
        stepSize = int(currentSampleRate / newSampleRate)
        interp = epoch[range(0, epoch.shape[0], stepSize)]
        return interp

    @staticmethod
    def removeBadChannels(data, channels, axis=1):
        r"""
        Remove channels from data along the specified exis

        Args:
            data (np.ndarray): The tensor
            channels (int list): list of bad channels
            axis (int): axis to zscore along. If this is None, then the overall zscore is given
        """
        return np.delete(data, channels, axis)

    @staticmethod
    def normaliseTarget(data, axis=0):
        r"""
        The target is normalized (z-scored) independently for each finger

        Args:
            data (np.ndarray): The tensor to be zscored
            axis (int): axis to zscore along. If this is None, then the overall zscore is given
        """
        out = np.zeros(data.shape)
        for i in range(0, data.shape[1]):
            out[:, i] =  zscore(data[:,i])
        return out
