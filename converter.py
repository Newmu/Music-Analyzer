import sh
import os

def Delete(filePath):
	'''
	Helper routines handeling sys calls for conversion/cleanup.
	'''
	sh.rm(filePath)

def Convert(filePath,format):
	'''
	Helper routines handeling sys calls for conversion/cleanup.
	'''
	newPath = '.'.join(os.path.splitext(filePath)[:-1]).strip()+format
	if os.path.exists(newPath):
		Delete(newPath)
	sh.avconv('-i',filePath,newPath)
	return newPath