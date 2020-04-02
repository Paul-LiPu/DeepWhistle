import wave
import numpy as np
import math
from python_speech_features import sigproc, base
import os
import fnmatch
import cv2
import wavio

# Spectrogram and ground truth generation.
def log_magnitute_spectrum_GT_DCL_blockwise3(processBlock_func, frame_time_span, step_time_span, imsave_output_dir, min_freq, max_freq, split_time, clip_min=0, clip_max=7):
    def plot_magnitute_spectrum(wav_bin):
        wav_data = get_wav_info_and_data2(wav_bin[0])
        if wav_data['samplewidth'] > 2:
            wav_data['data'] /= 2 ** (8 * (wav_data['samplewidth'] - 2))
        bin_data = read_tonal_file(wav_bin[1])
        split_func = split_into_frames_params_new(frame_time_span, step_time_span, wav_data['framerate'])
        freq_resolution = 1000 // frame_time_span
        spectrum_split = split_time // step_time_span
        wav_filename = os.path.basename(wav_bin[0])
        split_result = wav_filename.split('.wav')
        wav_filename = split_result[0]
        wav_time_ms = wav_data['nframes'] * 1000.0 / wav_data['framerate'] - step_time_span
        nsplit = int(math.ceil(wav_time_ms / split_time))
        for i in xrange(nsplit):
            start_time_ms = i * split_time
            end_time_ms =(i + 1) * split_time if (i + 1) * split_time < wav_time_ms else wav_time_ms
            imsave_output_filename = imsave_output_dir + '/' + wav_filename + '_uniclip_logspec' + str(i) + '.png'
            GT_output_filename = imsave_output_dir + '/' + wav_filename + '_uniclip_logspec' + str(i) + '_GT.png'
            spec_split, GT_split =  processBlock_func(wav_data, split_func, bin_data, freq_resolution, min_freq, max_freq,
                                                 clip_min, clip_max, step_time_span, start_time_ms, end_time_ms)
            norm_figure = spec_split
            norm_figure_flip = norm_figure[::-1, ]
            GT_split_flip = GT_split[::-1, ]
            cv2.imwrite(imsave_output_filename, (norm_figure_flip * 255))
            cv2.imwrite(GT_output_filename, (GT_split_flip * 255))
        return nsplit
    return plot_magnitute_spectrum


# Spectrogram and ground truth generation for one time block.
def processBlock_lineGT(wav_data, split_func, bin_data, freq_resolution, min_freq, max_freq, clip_min, clip_max, step_time_span, start_time, end_time):
    start_frame = int(start_time / 1000 * wav_data['framerate'])
    end_frame = int((end_time / 1000.0 + 1.0 / freq_resolution - step_time_span / 1000.0)* wav_data['framerate'])
    frames = split_func(wav_data['data'][start_frame:end_frame])
    NFFT = len(frames[0])
    singal_magspec = sigproc.magspec(frames, NFFT)
    clip_bottom = min_freq // freq_resolution
    clip_top = max_freq // freq_resolution + 1
    output_spec = singal_magspec.T[clip_bottom:clip_top]
    output_spec = np.log10(output_spec)
    output_spec = normalize3(output_spec, clip_min, clip_max)
    GT_spec = np.zeros(output_spec.shape)
    for record in bin_data:
        time_stamps = np.asarray(record['Time'])
        sort_index = np.argsort(time_stamps)
        freqs = np.asarray(record['Freq'])
        time_stamps = time_stamps[sort_index]
        freqs = freqs[sort_index]
        previous_bin = 0
        previous_freq = 0
        first_flag = True
        for i in xrange(len(time_stamps)):
            if time_stamps[i] < start_time / 1000 or time_stamps[i] > end_time / 1000:
                continue
            time_stamp = (time_stamps[i] - start_time / 1000) * 1000 / step_time_span
            freq = (freqs[i] - min_freq) / freq_resolution
            if first_flag:
                previous_bin = time_stamp
                previous_freq = freq
                first_flag = False
                continue
            freq_time_line = two_point_line(x1=previous_bin, y1=previous_freq, x2=time_stamp, y2=freq)
            for j in range(int(math.floor(previous_bin)), int(math.ceil(time_stamp)) + 1):
                if j >= GT_spec.shape[1]:
                    continue
                if time_stamp - previous_bin < 1e-10:
                    current_freq = freq
                    line_width = int(math.ceil(abs((previous_freq - freq) / 2)))
                else:
                    current_freq, line_width = freq_time_line(j)
                for t in range(int(math.floor(current_freq)), int(math.ceil(current_freq)) + line_width):
                    if t < 0 or t >= GT_spec.shape[0]:
                        continue
                    GT_spec[t, j] = 1
            previous_bin = time_stamp
            previous_freq = freq
            if freqs[i] < min_freq or freqs[i] > max_freq:
                continue
            if int(math.ceil(time_stamp)) >= GT_spec.shape[1]:
                continue
            GT_spec[int(math.floor(freq)), int(math.floor(time_stamp))] = 1
            GT_spec[int(math.ceil(freq)), int(math.floor(time_stamp))] = 1
            GT_spec[int(math.floor(freq)), int(math.ceil(time_stamp))] = 1
            GT_spec[int(math.ceil(freq)), int(math.ceil(time_stamp))] = 1
    return output_spec, GT_spec


# Get .wav file's information and data.
def get_wav_info_and_data2(wav_file):
    wf = wave.open(wav_file, "rb")
    nframes = wf.getnframes()
    framerate = wf.getframerate()
    nchannels = wf.getnchannels()
    samplewidth = wf.getsampwidth()
    wav_data = get_frames_data_from_filename(wav_file)
    result = {}
    result['nframes'] = nframes
    result['framerate'] = framerate
    result['nchannels'] = nchannels
    result['samplewidth'] = samplewidth
    result['data'] = wav_data.data.ravel()
    result['class'] = get_wav_class(wav_file)
    result['file'] = os.path.basename(wav_file)
    wf.close()
    return result

# find .wav belong to which class of animal
def get_wav_class(wav_file):
    parent_dir = os.path.dirname(wav_file)
    parent_folder = os.path.basename(parent_dir)
    return parent_folder

# Get .wav file's data.
def get_frames_data_from_filename(wav_file):
    wave_data = wavio.read(wav_file)
    return wave_data

# split raw wave signal to frames to do fft.
def split_into_frames_params_new(frame_time_span, step_time_span, framerate):
    def split_into_frames(wav_data):
        signal = wav_data
        frame_sample_span = int(math.floor(frame_time_span / 1000.0 * framerate) + 1)
        step_sample_span = int(math.floor(step_time_span / 1000.0 * framerate))
        if len(signal) < frame_sample_span:
            return []
        frames = sigproc.framesig(signal, frame_sample_span, step_sample_span)
        return frames
    return split_into_frames


# min-max normalization
def normalize3(mat, min_v, max_v):
    mat = np.clip(mat, min_v, max_v)
    return (mat - min_v) / (max_v - min_v)

# get wave samplewidth
def get_wav_samplewidth(wav_file):
    wf = wave.open(wav_file, "rb")
    wav_filename = os.path.basename(wav_file)
    wav_filename = wav_filename.split('.wav')[0]
    samplewidth = wf.getsampwidth()
    print(wav_filename + ' samplewidth: ' + str(samplewidth))
    wf.close()

# calculate another point on two points defined line.
def two_point_line(x1, y1, x2, y2):
    def line_func(x):
        if x1 - x2 > 1e-10:
            line_width = int(math.ceil(abs((y1 - y2)/(x1 - x2))))
        else:
            line_width = int(math.ceil(abs(y1 - y2) / 2))
        if y1 - y2 == 0:
            line_width = 1
        return (y1 - y2)/(x1 - x2)*(x - x2) + y2,line_width
    return line_func

####### Help functions for directory and file operation.

# find file  with certain pattern.
def findfiles(path, fnmatchex='*.*'):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnmatchex):
            fullname = os.path.join(root, filename)
            result.append(fullname)
    return result

# Get all subdirs in one directory
def list_all_dir(path):
    result = []
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path, file)
        if os.path.isdir(m):
            result.append(m)
    return result

# Get .wav files in one directory
def find_wav_files(path):
    return findfiles(path, fnmatchex='*.wav')


# check if dir exists. if not, create it.
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Substitue postfix .bin to .wav.
def bin2wav_filename(bin_file):
    bin_filename = os.path.basename(bin_file)
    bin_name, ext = os.path.splitext(bin_filename)
    return bin_name + '.wav'

####### Read .bin file helper function.
from tonal_lipu import tonal
def read_tonal_file(tonal_file):
    tonal_reader = tonal(tonal_file)
    return list(tonal_reader)
