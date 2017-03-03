#!/usr/bin/python

'''
Script to concatenate ATM icessn files into a single file and add fields for date and UTC

Usage: catATM.py <icessn_files>
'''

import sys, glob, time

#===============================================
# Functions
#===============================================
# Main

# Get icessn filenames that were passed as arguments
# infiles = ['ILATM2_20091031_142352_smooth_nadir5seg_50pt.csv','ILATM2_20091031_141955_smooth_nadir5seg_50pt.csv']
try:
	# filenames = [f for f in infiles if f.__contains__('_smooth_') if f.endswith('_50pt.csv')]
	filenames = [f for f in sys.argv[1:] if f.__contains__('_smooth_') if f.endswith('_50pt.csv')]
	filenames[0]
except:
	print __doc__
	exit()

output_filename = 'ATM_all_NewFormat.txt'
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
				# gpsTime = float(line.split()[0])
				gpsTime = line.split()[0]
				# If seconds of day roll over to next day
				if gpsTime < prevTime:
					date = str(int(date) + 1)

				# # Determine UTC offset from date
				# UTCoffset = getUTCoffset(int(date[:4]), int(date[4:6]), int(date[6:]))
				# # Calculate "normal-person time"
				# utc = gpsTime - UTCoffset
				# if utc < 0:
				# 	utc += 86400
				# utc = str(utc).split('.')
				# utc = utc[0] + '.' + utc[1].ljust(4, '0')
				# utc = gpsTime

				# Create new data record (with EOL in "line" variable)
				newline = '{0}, {1}'.format(date, line)
				print newline
				f.write(newline + '\n')
				# prevTime = gpsTime
