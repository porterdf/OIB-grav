#!/usr/bin/python

'''
Script to concatenate ATM icessn files into a single file and add fields for date and UTC

Usage: catATM.py <icessn_files>
'''

import sys, glob, time

#===============================================
# Functions
def getUTCoffset(yyyy, mm, dd):
	'''
	Calculate UTC offset from date (offsets defined below)
			GPS time = UTC time + UTCoffset(positive)
	'''
	o_yy = [1980, 1981, 1982, 1983, 1985, 1988, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1999, 2006, 2009, 2012]
	o_mm = [1, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 1, 7, 1, 1, 1, 7]
	o_dd = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	offsets = []
	for i in range(len(o_yy)):
		offsets.append((i, time.mktime((o_yy[i], o_mm[i], o_dd[i], 0, 0, 0, 0, 0, 0))))
	t = time.mktime((int(yyyy), int(mm), int(dd), 0, 0, 0, 0, 0, 0))
	UTCoffset = 'ERROR'
	for offset, ls_date in offsets:
		if ls_date > t: break
		UTCoffset = offset
	return UTCoffset

#===============================================
# Main

# Get icessn filenames that were passed as arguments
try:
	filenames = [f for f in sys.argv[1:] if f.__contains__('_smooth_') if f.endswith('_50pt')]
	filenames[0]
except:
	print __doc__
	exit()

output_filename = 'ATM_all.txt'
tiles = [0]

# Open output file
with open(output_filename, 'w') as f:

	# Loop through filenames
	for filename in filenames:
		print 'Extracting records from {0}...'.format(filename)
		# Get date from filename
		# date = '20' + filename[:6]	# this is Linky's original code
		date = filename[7:15]
		prevTime = 0

		# Loop through lines in icessn file
		for line in open(filename):
			# Make sure records have the correct number of words (11)
			if (len(line.split()) == 11) and (int(line.split()[-1]) in tiles):
				line = line.strip()
				gpsTime = float(line.split()[0])
				# If seconds of day roll over to next day
				if gpsTime < prevTime:
					date = str(int(date) + 1)

				# Determine UTC offset from date
				UTCoffset = getUTCoffset(int(date[:4]), int(date[4:6]), int(date[6:]))
				# Calculate "normal-person time"
				utc = gpsTime - UTCoffset
				if utc < 0:
					utc += 86400
				utc = str(utc).split('.')
				utc = utc[0] + '.' + utc[1].ljust(4, '0')

				# Create new data record (with EOL in "line" variable)
				newline = '{0} {1} {2}'.format(date, utc, line)
				print newline
				f.write(newline + '\n')
				prevTime = gpsTime
