import eyed3
from pprint import pprint

class Meta(object):
	def __init__(self,mp3Data):
		self.artist = mp3Data.tag.artist
		self.album = mp3Data.tag.album
		self.title = mp3Data.tag.title
		self.track_num = mp3Data.tag.track_num
		self.genre = mp3Data.tag.genre.name
		self.date = mp3Data.tag.best_release_date.year

	def __str__(self):
		return str(self.__dict__)

# Extracting mp3 id3 tag data
def metaData(filePath):
	try:
		mp3Data = eyed3.load(filePath)
		metaData = Meta(mp3Data)
		return metaData
	except:
		print 'no metaData detected'
		return None