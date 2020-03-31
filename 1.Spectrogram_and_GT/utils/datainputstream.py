#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:36:27 2017

@author: kpalmer
"""


"""
Reading from Java DataInputStream format.
From https://github.com/arngarden/python_java_datastream
This uses big endian (network) format.
"""

import struct

class DataInputStream:
    def __init__(self, stream):
        self.stream = stream
        
    def read_record(self, format, n=1, order=">"):
        """read_record(format, n)
        format - format of record (e.g. 'dd', see structu.unpack)
        n - number of records
        order - order ">" network/big-endian (default) or "<" little endian
        """
        
        # Build up format string and figure out how  large it is
        unpackstr = order + format * n
        bytes = struct.calcsize(unpackstr)
        
        data = self.stream.read(bytes)
        if len(data) < bytes:
            if len(data) == 0:
                raise EOFError  # no data, end of file
            else:
                # We didn't get all the data we expected, but we got some
                raise ValueError("Expected {} bytes, read {} before EOF".format(
                    bytes, len(data)))
        
        values = struct.unpack(unpackstr, data)
        return values
                            
    def read_n_doubles_and_bools(self, ndoubles=0, nbools=0, n=1):        
        unpack_fmt_string =  'd'*ndoubles + '?'*nbools
        return self.read_record(unpack_fmt_string, n)

    def read_boolean(self):
        return self.read_record("?")[0]

    def read_byte(self):
        return self.read_record("b")[0]

    def read_unsigned_byte(self):
        return self.read_record("B")[0]

    def read_char(self):
        return self.read_record("H")[0]

    def read_double(self):
        return self.read_record("D")[0]

    def read_float(self):
        return self.read_record("f")[0]

    def read_short(self):
        return self.read_record("h")[0]

    def read_unsigned_short(self):
        return self.read_record("H")[0]

    def read_long(self):
        return self.read_record("q")[0]

    def read_utf(self):
        "read_utf - Read "
        utf_length = struct.unpack('>H', self.stream.read(2))[0]
        return self.stream.read(utf_length)

    def read_int(self):
        return self.read_record("i")[0]
  
    def read_unsigned_int(self):
        return self.read_record("I")[0]
