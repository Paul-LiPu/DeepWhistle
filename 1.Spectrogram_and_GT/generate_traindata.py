#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-19 5:34 pm 
# @Author  : Pu Li
# @File    : generateSpectrogram.py

import os
import utils.wav2spec as wav2spec

# Test data file timestamps. 
test_dict = ('154040','171606','165000','100318','100219','125009','121518','205305','205730')


# base_dir = '/home/lipu/Documents/whale_recognition'
tonal_file = '/media/lipu/SAUMIL/SilbedoAnnotationsdclde2013/2009_0328/Ch. 1/NOPP6_EST_20090328_000000_c01.ann'

tonal_reader = wav2spec.read_tonal_file(tonal_file)


# # collect all .wav files
# wav_dir = base_dir + '/Train_raw_data/MarineAccousticWorkshop/Data/5th_DCL_data_common'
# wav_dir2 = base_dir + '/Train_raw_data/MarineAccousticWorkshop/Data/5th_DCL_data_bottlenose'
# wav_files = wav2spec.find_wav_files(wav_dir) + wav2spec.find_wav_files(wav_dir2)
#
# # collect all .wav filenames
# wav_names = map(os.path.basename, wav_files)
# wav_file_dict = {wav_names[i] : wav_files[i] for i in range(len(wav_names))}
#
# # collect all .bin files.
# anno_dir = base_dir + '/Train_raw_data/MarineAccousticWorkshop/Annotations/common'
# anno_dir2 = base_dir + '/Train_raw_data/MarineAccousticWorkshop/Annotations/bottlenose'
# exp_group = os.path.basename(anno_dir)
# bin_files = wav2spec.findfiles(anno_dir, fnmatchex='*.bin') + wav2spec.findfiles(anno_dir2, fnmatchex='*.bin')
# temp = []
#
# def check2(filename, test_dict):
#     for test_num in test_dict:
#         if test_num in filename:
#             return True
#     return False
#
# # find all .bin files with timestamps in test_dict
# for file in bin_files:
#     filename = os.path.basename(file)
#     if check2(filename, test_dict):
#         temp.append(file)
#
# # find all .wav file paths with selected .bin files.
# bin_files = temp
# anno_wav_filenames = map(wav2spec.bin2wav_filename, bin_files)
#
# anno_wav_files = [wav_file_dict[filename] for filename in anno_wav_filenames]
# map(wav2spec.get_wav_samplewidth, anno_wav_files)
#
# ### generate all spectrum
# ## parameter setting
# frame_time_span = 8 # ms, length of time for one time window to do dft.
# step_time_span = 2 # ms, length of time step.
# clip_min = 0
# clip_max = 6 # log magnitude spectrogram min-max normalization parameter
# min_freq = 5000 # Hz, lower bound of frequency for spectrogram
# max_freq = 50000 # Hz, upper bound of frequency for spectrogram
# split_time = 3000 # ms, length of time for each output spectrogram image.
# output_folder = exp_group + '_framel-' + str(frame_time_span) + '_step-' + str(step_time_span) + '_log_magspec_wavio_24bit_block_lineGT_test1'
# imsave_output_dir = base_dir + '/Train_data/' + output_folder
# wav2spec.check_dir(imsave_output_dir)
#
# # universal normalized magnitute spectrum
# count = 0
# files = [0]
for i in range(0, len(anno_wav_filenames)):
    print i
    wav_file = anno_wav_files[i]
    wav_filename = os.path.basename(wav_file)
    wav_filename = wav_filename.split('.wav')[0]
    block_func = wav2spec.processBlock_lineGT
    output_dir = imsave_output_dir + '/' + wav_filename
    wav2spec.check_dir(output_dir)
    output_func = wav2spec.log_magnitute_spectrum_GT_DCL_blockwise3(block_func, frame_time_span=frame_time_span, step_time_span=step_time_span, imsave_output_dir=output_dir,
                                         min_freq=min_freq, max_freq=max_freq, split_time=split_time, clip_min=clip_min, clip_max=clip_max)
    count = output_func([anno_wav_files[i], bin_files[i]])
    print 'number of output: ' + str(count)


