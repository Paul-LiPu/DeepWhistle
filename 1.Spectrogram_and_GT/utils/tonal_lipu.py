#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-18 下午11:21
# @Author  : Pu Li
# @File    : tonal_lipu.py
from datainputstream import DataInputStream
import os
import numpy as np

class TonalHeader():
    # initial header string, what it should say if there is a proper header
    HEADER_STR = "silbido!"

    # Constant set to 3
    DET_VERSION = 3

    # construct the bitmask for each of the feature columns
    # t=1, f=2, snr=3 and so on
    TIME = 1
    FREQ = 1 << 1  # 2 because we are in binary
    SNR = 1 << 2
    PHASE = 1 << 3
    SCORE = 1 << 4
    CONFIDENCE = 1 << 5
    RIDGE = 1 << 6
    # Default bitmask indicating variables present
    DEFAULT = TIME | FREQ

    def __init__(self, ton_filename, ):
        "init((filename) - construct silbidio tonal header reader"

        # Define filename
        self.ton_filename = ton_filename

        # number of bytes to read in to check if header is present
        self.magicLen = len(self.HEADER_STR)

        # load the file
        self.binary = open(self.ton_filename, "rb")

        # data input stream
        self.datainstream = DataInputStream(self.binary)

        # this is a function...
        self.ReadHeader()

    def ReadHeader(self):

        # Read in the first 8 bytes of the file-
        headerlabel = self.binary.read(self.magicLen)

        # If there is a header, set things up approperiately
        if headerlabel == self.HEADER_STR:

            # set up stream reader and then read appropriate sizes
            self.version = self.datainstream.read_short()  # Use right one
            self.bitMask = self.datainstream.read_short()
            self.userVersion = self.datainstream.read_short()
            self.headerSize = self.datainstream.read_int()

            # Figure out how much of the header has already been read
            self.headerused = 2 + 2 + 2 + 4 + self.magicLen  # Length read in up till now in bytes

            ## Figure out how long the user comments must be
            commentLen = self.headerSize - self.headerused

            if (commentLen > 0):
                self.comment = self.datainstream.read_utf()
            else:
                self.comment = '';


        else:  # no header
            self.bitMask = self.DEFAULT
            # set pointer back to byte 0 from datainpustream modification
            self.binary.seek(0)
            # print(self.bitMask)
            pass

    def hasSNR(self):
        return bool((self.bitMask & self.SNR) > 0)

    def hasPHASE(self):
        return bool((self.bitMask & self.PHASE) > 0)

    def hasRIDGE(self):
        return bool((self.bitMask & self.RIDGE) > 0)

    def hasFREQ(self):
        return bool((self.bitMask & self.FREQ) > 0)

    def hasTIME(self):
        return bool((self.bitMask & self.TIME) > 0)

    def hasCONFIDENCE(self):
        return bool((self.bitMask & self.CONFIDENCE) > 0)

    def hasSCORE(self):
        return bool((self.bitMask & self.SCORE) > 0)

    # get some things
    def getComment(self):
        return str(self.comment)

    def getUserVersion(self):
        return self.userVersion

        # COMMENTS REQIRED

    def getDatainstream(self):
        "getDataInstream() - Return DataInputStream that accesses file"
        return self.datainstream

    """
    def getFileFormatVersion(self):
        return self.comment
    """

    def getMask(self):
        return self.bitMask



class tonal(object):
    # Initialize values
    def __init__(self, fname, ID=0, Time=0, Freq=0, Ntonals=0, verbose=False):
        "__init__(filename, debug)"

        self.verbose = verbose
        self.whistle_idx = 0  # keep track of current whistle

        self.fname = fname
        self.hdr = TonalHeader(fname)  # use the tonal header
        self.Time = Time
        self.Freq = Freq
        self.Ntonals = Ntonals
        self.binary = open(self.fname, "rb")  # open the binary file
        self.curent = 0  # starting place for the iterator
        self.ID = ID  # tonal ID (1 ->N)
        self.SNR = None
        self.Phase = None
        self.Score = None
        self.Confidence = None
        self.Ridge = None

        # set up the optional variables
        # SNR
        if self.hdr.hasSNR():
            self.SNR = 0
        # Phase
        if self.hdr.hasPHASE():
            self.Phase = 0
        # Score
        if self.hdr.hasSCORE():
            self.Score = 0
        # Confidence
        if self.hdr.hasCONFIDENCE():
            self.Confidence = 0
        # Ridge
        if self.hdr.hasRIDGE():
            self.Ridge = 0

        # Define data input stream
        self.bis = self.hdr.getDatainstream()

        # Set read format for whistle time-frequency nodes
        self.timefreq_fmt = 'dd' + self.hdr.hasSNR() * 'd' + \
                            self.hdr.hasPHASE() * 'd' + self.hdr.hasCONFIDENCE() * 'i' + \
                            self.hdr.hasRIDGE() * 'i'

    '''    
    # Check to see whether there is a header and offset by the number of bytes in the header
    # if the bitmask is equal to 3  then only time and frequency were provided and
    # no offset??

    if self.hdr.getMask() == 3:
        # Read in the files! 
    else:
        print('Header present, you are stuffed')
    break
        # offset by the length of the header and then read the files!

    '''

    # Define tonal as an iteratable object - THERE ARE THINGS IN HERE YOU CAN
    # ITERATE
    def __iter__(self):
        "iter(obj) - Return self as we know how to iterate"
        return self

    def __next__(self):
        'next() - Return next whistle'

        # set a dictionary for the whistle
        keyDict = {"Time", "Freq", "SNR", "Phase", "Ridge"}
        keylist = ["Time", "Freq", "SNR", "Phase", "Ridge"]
        Whistle_contour = dict([(key, []) for key in keyDict])

        if self.verbose:
            print("Reading whistle {} in file {}".format(
                self.whistle_idx, self.fname))
        try:
            NumNodes = self.bis.read_int()
        except EOFError:
            raise StopIteration  # No more whistles in file

        if self.verbose:
            print(", {} nodes".format(NumNodes))

        # Read time-frequency nodes associated with whistle
        data = self.bis.read_record(format=self.timefreq_fmt, n=NumNodes)

        # Throw a warning if whistle has noting in it
        if len(data) < 1:
            print_msg = 'Problem with ' + \
                        os.path.split(os.path.split(self.fname)[0])[1] + ' ' + \
                        os.path.split(self.fname)[1] + ' no data read'

            print(print_msg)

        n_metrics = 2 + bool(self.SNR is not None) + bool(self.Phase is not None)
        for ii in range(n_metrics):
            key = keylist[ii]
            Whistle_contour[key] = data[ii::n_metrics]

        self.whistle_idx += 1
        if len(Whistle_contour['Time']) < 1:
            aa = self.fname + 'error!'
            print(aa)

        return Whistle_contour

    # getters
    def getFname(self):
        return str(self.fname)

    def getTime(self):
        return np.array(self.Time)

    def getFreq(self):
        return np.array(self.Freq)

    def getSNR(self):
        if not self.hdr.hasSNR():
            print('Tonal has no SNR values')

    def getPhase(self):
        if not self.hdr.hasPHASE():
            print('Tonal has no Phase values')

    def getScore(self):
        if not self.hdr.hasSCORE():
            print('Tonal has no Score values')

    def getConf(self):
        if not self.hdr.hasCONFIDENCE():
            print('Tonal has no confidence values')

    def getRidge(self):
        if not self.hdr.hasRIDGE():
            print('Tonal has no Ridge values')

    def next(self):
        return self.__next__()


# tonal_file = '/home/sensetime/data/projects/whale_recognition/Train_raw_data/MarineAccousticWorkshop/Annotations/melon-headed/palmyra102006-061020-204454_4.bin'
# print os.path.exists(tonal_file)
# tonal_reader = tonal(tonal_file)
# tonals = []
# for t in tonal_reader:
#     tonals.append(t)
# temp = 1