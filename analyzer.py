'''
Extracts meta data and feature vectors from the specified song, prints results to console.
Dependencies are sh,eyed3,numpy,scipy,pylab

Example usage is: python analyzer.py digitalLove.mp3

Segments is a list of segment objects containg feature vectors for each segment of the song.

Format is:

3 descriptors - start end duration
4 loudness vectors - start,end,max (db), and max_time (within segment)
12 pitch vectors (0-1 normalized)
25 timbre vectors (db)

Based off http://web.media.mit.edu/~tristan/phd/dissertation/chapter3.html

Missing spectral masking implementation.

Should switch to a basis function based timbre descriptor in the future
'''

import featureExtractor as FE
import metaExtractor as ME
import os
import sys

class Analyzer(object):
	def __init__(self,filePath):
		print 'analyzing',filePath
		if os.path.splitext(filePath)[-1] == '.mp3':
			self.segments = FE.segmentData(filePath)
			self.meta = ME.metaData(filePath)
		else:
			raise TypeError('Not a .mp3!')

def main():
	song = Analyzer(sys.argv[1])
	print song.meta
	segment = song.segments[0]
	print segment.start
	print segment.end
	print segment.duration
	print segment.loudness
	print segment.pitches
	print segment.timbre

if __name__ == "__main__":
    main()