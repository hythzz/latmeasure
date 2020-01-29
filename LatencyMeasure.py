import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import wavfile

# config
debug = False       
simpletest = False
ts = 0.0005         # step in second
recordTime = 3.0    # time in second
blocksize= 2048     # block size for playback audio


class LatencyMeasure:
    def __init__(self):
        self.playbackIdx = 0
        self.recordIdx = 0
        self.inputData = np.zeros((0, 2))
        self.anchorData = np.zeros((0, 2))
        self.anchorFile = "Asset/anchor.wav"
        self.testFile = "Asset/500ms_off.wav"
        self.expTestLatency = 500               # in ms


    def readfile(self, filename):
        fs, data = wavfile.read(filename)
        return data, fs


    def freqResponse(self, data, fs):
        # only working on single channel
        data = data[:,0]

        # fft, only consider norm of complex numbers
        frdata = np.fft.fft(data) # only take one channel of audio
        absfr = np.absolute(frdata)

        # normallization based ratio of actual datapoints to fs
        dpCount = data.shape[0]
        freq = np.arange(dpCount)
        scale = (dpCount / fs)
        freq = freq / scale
        if scale != 1.0:
            # make every FR same # of data points, useful for later computation
            f = interpolate.interp1d(freq, absfr)
            freq = np.arange(fs)
            absfr = f(freq)
        absfr /= fs
        if debug:
            plt.figure()
            plt.plot(freq, absfr)
            plt.savefig("signal_fft.pdf")
        return absfr


    def powerSpectralDensity(self, fft, fs):
        psd = (1/(fs)) * (np.abs(fft) ** 2)
        db = 10 * np.log10(psd)
        maxdb = np.max(db)
        if debug:
            print("maxdb decibel: ", maxdb)
            freq = range(0, fs)
            plt.figure()
            plt.plot(freq, db)
            plt.savefig("signal_psd.pdf")
        return maxdb


    def isSameSignal(self, data1, data2, fs):
        anchorFR = self.freqResponse(data1, fs)
        dataFR = self.freqResponse(data2, fs)
        maxdb = self.powerSpectralDensity(dataFR - anchorFR, fs)
        return maxdb <= 0


    def signalDiff(self, anchor, data, fs):
        # only working on single channel
        anchor = anchor[:,0]
        data = data[:,0]

        window = len(anchor)
        step = int(fs / (1 / ts))
        offsets = range(0, len(data) - window, step)

        meandiff = np.asarray([np.sum(np.abs(data[idx:idx+window] - anchor)) / fs for idx in offsets])
        idx = np.argmin(meandiff)
        if debug:
            x = np.arange(len(meandiff))
            plt.plot(x, meandiff)
            plt.savefig("signal_diff.pdf")
        return idx


    def Test(self):
        print("Running Simple Test...")
        print("Anchor Audio file:\t{}".format(self.anchorFile))
        print("offset Audio file:\t{}".format(self.testFile))
        print("Latency suppose to be\t{} ms".format(self.expTestLatency))

        #two files prepared assumed has same sampling rate
        self.anchorData, fs = self.readfile(self.anchorFile)
        self.inputData, _ = self.readfile(self.testFile)

        if not self.isSameSignal(self.anchorData, self.inputData, fs):
            print("Not the Same Signal received, wrong input/output device?")
            exit(-1)

        idx = self.signalDiff(self.anchorData, self.inputData, fs)
        delay = idx / (1 / ts) * 1000
        print("Measured latency is\t{:.1f} ms".format(delay))


    def audioCallback(self, indata, outdata, frames, time, status):
        global blocksize
        if status:
            print(status)
        
        # output
        if len(self.anchorData) <= self.playbackIdx:
            outdata[:] = np.zeros((len(outdata), 2), dtype=np.int16)
        elif len(self.anchorData) >= self.playbackIdx + blocksize:
            outdata[:] = self.anchorData[self.playbackIdx:self.playbackIdx+blocksize]
            self.playbackIdx += blocksize
        elif len(self.anchorData) < self.playbackIdx + blocksize:
            end = len(self.anchorData) - self.playbackIdx
            outdata[:end] = self.anchorData[self.playbackIdx:]
            outdata[end:] = np.zeros((len(outdata) - end, 2), dtype=np.int16)
            self.playbackIdx = len(self.anchorData)

        # input
        rest = len(self.inputData) - self.recordIdx
        if len(indata) <= rest:
            self.inputData[self.recordIdx:self.recordIdx+len(indata),:] = indata
            self.recordIdx += len(indata)
        else:
            self.inputData[self.recordIdx:,:] = indata[:rest,:]
            self.recordIdx = len(self.inputData)

        if self.recordIdx == len(self.inputData):
            sd.CallbackStop()


    def Measure(self):
        global blocksize
        print("Running Measurement...")

        self.anchorData, fs = self.readfile(self.anchorFile)
        self.inputData = np.zeros((int(recordTime * fs), 2), dtype=np.int16)
        stream = sd.Stream(samplerate=fs, blocksize=blocksize, channels=2,
                           dtype=np.int16, callback=self.audioCallback)
        with stream:
            sd.sleep(int(recordTime * 1000 + 100))

        # Normalize power of input signal
        ratio = np.max(self.anchorData) / np.max(self.inputData)
        inputData = np.multiply(self.inputData, ratio)

        if debug:
            wavfile.write('input.wav', fs, inputData)
            anchortime = range(len(self.anchorData))
            plt.figure()
            plt.plot(anchortime, self.anchorData)
            plt.savefig("signal_anchor.pdf")
            time = range(len(self.inputData))
            plt.figure()
            plt.plot(time, self.inputData)
            plt.savefig("signal_input.pdf")

        if not self.isSameSignal(self.anchorData, self.inputData, fs):
            print("Not the Same Signal received, wrong input/output device?")
            exit(-1)

        idx = self.signalDiff(self.anchorData, self.inputData, fs)
        delay = idx / (1 / ts) * 1000
        print("Measured latency is\t{:.1f} ms".format(delay))



if __name__ == '__main__':
    if debug:
        print(sd.query_devices())

    measureObj = LatencyMeasure()
    if simpletest:
        measureObj.Test()
    else:
        measureObj.Measure()
