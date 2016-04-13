"""
Generic object pickler and compressor.

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""

import cPickle as pickle
import gzip


def save(object, filename, bin = 1):
	"""
	Saves a compressed object to disk.
	"""
	file = gzip.GzipFile(filename, 'wb')
	file.write(pickle.dumps(object, bin))
	file.close()


def load(filename):
	"""
	Loads a compressed object from disk.
	"""
	file = gzip.GzipFile(filename, 'rb')
	buffer = ""
	while 1:
		data = file.read()
		if data == "":
			break
		buffer += data
	object = pickle.loads(buffer)
	file.close()
	return object

