'''
Segments audio and extracts feature vectors for each segement
Returns a list of those segment objects
Use by calling Segement(filePath)
'''

import numpy as np 
import scipy.io.wavfile as sciowav
from pylab import specgram
import sys
import converter
import sh
import os

class Segment(object):
	def __init__(self,seg,timbre,pitches,loudness):
		self.start = seg[0]
		self.end = seg[1]
		self.duration = seg[1]-seg[0]
		self.loudness = loudness
		self.timbre = timbre
		self.pitches = pitches

	def __str__(self):
		return str(self.__dict__)

class Loudness(object):
	def __init__(self,DBSegLoudness,binTime):
		self.start = DBSegLoudness[0]
		self.end = DBSegLoudness[-1]
		self.max = np.max(DBSegLoudness)
		self.max_time = np.argmax(DBSegLoudness)*binTime

	def __str__(self):
		return str(self.__dict__)

# Normalize a convolution kernel to sum to 1
# Prevents scaling of data
def normKernel(kernel):
	total = np.sum(kernel)
	kernel = kernel*(1/total)
	return kernel

# Convert array from decibels to power spectrum
def DBToPow(db):
	db = np.array(db)
	return np.power(10,(db/20))*60

# Convert array from power spectrum to decibels
def powToDB(powData):
	powData = np.array(powData)
	DB = 20*np.log10(powData/60)
	return np.clip(DB,-60,sys.float_info.max)

# Return a slice of power spectrum data for a segment
def getSegData(data,seg,bins):
	start = np.where(bins==seg[0])[0]
	end = np.where(bins==seg[1])[0]
	return data[:,start:end]

# Calculate bark values for frequency bands
def barkScale(f):
	a = 13*np.arctan(0.00076*f)
	b = 3.5*np.arctan(np.power((f/7500),2))
	return np.floor(a+b)

# Calculate the normalization values to mimic human perception of spectrogram frequencies
def DBNorm(f):
	f = f/1000;
	a = -3.64*np.power(f,-0.8)
	b = 6.5*np.exp(-0.6*np.power((f-3.3),2))
	c = np.power(f,4)/1000
	result = a+b-c
	result[0] = -150
	return result

# Normalizes specGram output for human frequency preception
def specGramAdjust(data,freqs):
	DBNorms = DBNorm(freqs)
	DBNorms = np.clip(np.array(DBNorm(freqs).reshape(-1,1)),-60,sys.float_info.max)
	data = powToDB(data)
	return DBToPow(data+DBNorms)

# Normalizes an FFT spectrum for human frequency preception
def FFTAdjust(data,freqs):
	DBNorms = DBNorm(freqs)
	DBNorms = np.clip(np.array(DBNorm(freqs)),-60,sys.float_info.max)
	data = powToDB(data)
	return data+DBNorms

# Evaluates guassian function to find weightings for points
def evalGaussian(vals,sigma):
	num = np.exp(-(np.power(vals,2)/(2*sigma*sigma)))
	denom = np.sqrt(2*np.pi)*sigma
	return num/denom

# If input is not a .wav convert to .wav
# Returns tuple of filePath to analyze and whether or not it converted
def handleConversion(filePath):
	converted = False
	if os.path.splitext(filePath)[-1] != '.wav':
		filePath = converter.Convert(filePath,'.wav')
		converted = True
	return filePath,converted

# Create list of loudness objects for each segment
def loudnessSegments(powerData,bins,length,segments):
	loudnessFeats = []
	binTime = length/bins.size
	for seg in segments:
		segPowerData = getSegData(powerData,seg,bins)
		segLoudness = []
		for i in xrange(segPowerData.shape[1]):
			segLoudness.append(np.mean(segPowerData[:,i]))
		DBSegLoudness = powToDB(segLoudness)
		loudness = Loudness(DBSegLoudness,binTime)
		loudnessFeats.append(loudness)
	return loudnessFeats

# Get timbre vectors for each segment
def timbreSegments(powerData,bins,segments,freqs):
	barks = barkScale(freqs)
	uniqueBarks = np.unique(barks)
	timbreVecs = []
	for seg in segments:
		segPowerData = getSegData(powerData,seg,bins)
		segTimbreVecs = []
		for i in xrange(segPowerData.shape[1]):
			timbreVec = []
			for bark in uniqueBarks:
			    timbreVec.append(segPowerData[:,i][barks == bark].mean())
			segTimbreVecs.append(timbreVec)
		timbreVecs.append(np.mean(segTimbreVecs,0))
	return powToDB(timbreVecs)

# Temporal masking of the audio spectrogram to mimic human perception
# Convolves with a 0.2 sec half-hann window
def tempMask(data,freqs,length,bins):
	dataMasked = np.zeros(data.shape)
	for i in xrange(freqs.size):
		winSize = round(0.4/(length/bins.size))
		hann = np.hanning(winSize)
		halfPoint = np.argmax(hann)
		hann = hann[halfPoint:]
		hann = normKernel(hann)
		dataMasked[i,:] = np.convolve(data[i,:],hann,mode='same')
	return dataMasked

# Segments the audio into subsections for each note or spectral change
# Calculates spectral varience and smooths result with hann win
# Detects peaks in spectral varience as segments
def getSegments(data,freqs,length,bins):
	specVar = np.zeros(data[0,:].shape)
	for i in xrange(freqs.size):
		specVar += np.abs(np.gradient(data[i,:]))
	winSize = round(0.15/(length/bins.size))
	hannWin = np.hanning(winSize)
	specVar = np.convolve(specVar,hannWin,mode='same')
	specVarGrad = np.gradient(specVar)
	onsets = [bins[i] for i in xrange(len(specVarGrad)-1) if specVarGrad[i] > 0 and specVarGrad[i+1] < 0]
	segments = []
	for i,onset in enumerate(onsets):
		if i == 0:
			segments.append([bins[0],onset])
		elif i == len(onsets)-1:
			segments.append([onset,bins[-1]])
		else:
			segments.append([onset,onsets[i+1]])
	return np.array(segments)

# Computes FFT of each segment
def FFTSegment(audio,Fs,seg):
	audioSeg = audio[seg[0]:seg[1]]
	n = len(audioSeg)+8096
	FFTSeg = 2*np.abs(np.fft.rfft(audioSeg,n))
	freqs = np.fft.fftfreq(n,float(1)/Fs)
	halfPoint = np.argmin(freqs)
	freqs = freqs[:halfPoint+1]
	FFTSeg = FFTAdjust(FFTSeg,freqs)
	FFTSeg = DBToPow(FFTSeg)
	FFTSeg = FFTSeg[1:-1]
	return FFTSeg,freqs

# Gets FFT of segment
# Shifts to equal tempermant (piano key) scale
# For every key on scale, computes intensity using
# Guassian windows of the power spectrum of key width
# Folds keys down to 12 note pitch scale
# Normalizes pitch scale to 0 to 1
def pitchSegments(audio,Fs,segments):
	segments = np.round(segments*44100) # Convert to array indices
	pitchScales = []
	avgs = []
	for seg in segments:
		FFTSeg, freqs = FFTSegment(audio,Fs,seg)
		keys = 12*np.log2(freqs/440)+49
		keys = keys[1:-1]
		pitchScale = np.zeros(12)
		for key in xrange(16,100):
			positions = keys[(keys > key-0.5) & (keys < key+0.5)]-key
			FFTVals = FFTSeg[(keys > key-0.5) & (keys < key+0.5)]
			weights = evalGaussian(positions,0.2)
			weights = normKernel(weights)
			index = (key-16)%12
			pitchScale[index] += np.sum(FFTVals*weights)
		pitchScale = pitchScale/np.max(pitchScale)
		pitchScales.append(pitchScale)
	return pitchScales

# Putting everything together
def merge(segs,timbres,pitches,loudness):
	segments = []
	for i in xrange(len(segs)):
		seg = Segment(segs[i],timbres[i],pitches[i],loudness[i])
		segments.append(seg)
	return segments

# Main function gets specgram data, process it for human perception
# Then does feature vector extraction.
def segmentData(filePath):
	filePath,converted = handleConversion(filePath)
	audio = sciowav.read(filePath)
	Fs = audio[0]
	npA = audio[1][:,0]
	length = float(npA.shape[0])/Fs
	powerData, freqs, bins, im = specgram(npA, NFFT=2048, Fs=Fs, noverlap=1536)
	powerData = specGramAdjust(powerData,freqs)
	maskedData = tempMask(powerData,freqs,length,bins)
	segments = getSegments(maskedData,freqs,length,bins)
	timbreSegs = timbreSegments(maskedData,bins,segments,freqs)
	pitchSegs = pitchSegments(npA,Fs,segments)
	loudSegs = loudnessSegments(maskedData,bins,length,segments)
	segments = merge(segments,timbreSegs,pitchSegs,loudSegs)
	if converted: 
		converter.Delete(filePath)
	return segments