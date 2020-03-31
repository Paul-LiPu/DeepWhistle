%% parameters
clear all;
addpath('./functions/')

% directory to training data and testing data. 
data_group = 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT';
test_group = 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_TEST';


% output directory
output_dir='./test/';
output_dir = [output_dir, '/', data_group];
if ~exist(output_dir)
    mkdir(output_dir);
end

%% generate training data. 
folder = ['/home/lipu/Documents/whale_recognition/Train_data/DCL/magspec_plot_original_all/', data_group, '/'];
subdirs = dir(fullfile(folder, '*'));

patchsize = 50;
stride = 25;
count = 0;
data = zeros(patchsize, patchsize, 1, 1);
label = zeros(patchsize, patchsize, 1, 1);
output_size = 200000;
count_total = 0;
file_num = 1;
for n = 3 : length(subdirs)
    subdir = subdirs(n).name;
    subdir = [folder, subdir,'/'];
    filepaths = dir(fullfile(subdir,'*GT.png'));
    for i = 1 : length(filepaths)
        disp([num2str(i / length(filepaths))])
        label_name = filepaths(i).name;
        temp = strsplit(label_name, '_GT');
        input_name = [temp{1}, '.png'];
        input_file = [subdir, input_name];
        image = imread(input_file,'png');
        image = image(:,:,1);
        image = single(image) / 255;
        
        [h, w] = size(image);
        
        label_file = [subdir, label_name];
        image_label = imread(label_file,'png');
        image_label = image_label(:,:,1);
        image_label = single(image_label) / 255;
        
        judge = sum(sum(image_label));
        disp(input_name);
        disp(label_name);
        disp(['number of whistles : ', num2str(judge)])
        if sum(sum(image_label)) == 0
            continue
        end
        [h_l, w_l] = size(image_label);
        h = min(h, h_l);
        w = min(w, w_l);
        for x = 1 : stride : h - patchsize + 1
            for y = 1 : stride : w - patchsize + 1
                data(:,:,1,count + 1) = image(x:x+patchsize-1, y:y+patchsize-1);
                label(:,:,1,count + 1) = image_label(x:x+patchsize-1, y:y+patchsize-1);
                count = count + 1;
                count_total = count_total + 1;
            end
        end
        if (count > output_size)
            order = randperm(count);
            data = data(:,:,:, order);
            label = label(:,:,:,order);
            savepath = [output_dir, '/train_', data_group,'_patch', num2str(patchsize),'_stride', num2str(stride), '_', num2str(file_num), '.h5'];
            chunksz = 128;
            created_flag = false;
            totalct = 0;
            for batchno = 1: floor(size(data, 4) / chunksz)
                last_read=(batchno-1)*chunksz;
                batchdata = data(:, :, :, last_read+1:last_read+chunksz);
                batchlabs = label(:, :, :, last_read+1:last_read+chunksz);
                
                startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
                curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
                created_flag = true;
                totalct = curr_dat_sz(end);
            end
            h5disp(savepath);
            data = data(:, :, :, totalct + 1 : end);
            label = label(:, :, :, totalct + 1 : end);
            count = count - totalct;
            file_num = file_num + 1;
        end
    end
    disp(['number of data : ', num2str(count)])
    disp(['number of total data : ', num2str(count)])
    if (count > output_size || n == length(subdirs))
        order = randperm(count);
        data = data(:,:,:, order);
        label = label(:,:,:,order);
        savepath = [output_dir, '/train_', data_group,'_patch', num2str(patchsize),'_stride', num2str(stride), '_', num2str(file_num), '.h5'];
        chunksz = 128;
        created_flag = false;
        totalct = 0;
        for batchno = 1: floor(size(data, 4) / chunksz)
            last_read=(batchno-1)*chunksz;
            batchdata = data(:, :, :, last_read+1:last_read+chunksz);
            batchlabs = label(:, :, :, last_read+1:last_read+chunksz);
            
            startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
            curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
            created_flag = true;
            totalct = curr_dat_sz(end);
        end
        h5disp(savepath);
        data = data(:, :, :, totalct + 1 : end);
        label = label(:, :, :, totalct + 1 : end);
        count = count - totalct;
        file_num = file_num + 1;
    end
end


%% generate testing data. 
folder = ['/home/lipu/Documents/whale_recognition/Train_data/DCL/magspec_plot_original_all/', test_group, '/'];
subdirs = dir(fullfile(folder, '*'));

patchsize = 50;
stride = 25;
count = 0;
data = zeros(patchsize, patchsize, 1, 1);
label = zeros(patchsize, patchsize, 1, 1);
for n = 3 : length(subdirs)
    subdir = subdirs(n).name;
    subdir = [folder, subdir,'/'];
    filepaths = dir(fullfile(subdir,'*GT.png'));
    for i = 1 : 2 : length(filepaths)
        disp([num2str(i / length(filepaths))])
        label_name = filepaths(i).name;
        temp = strsplit(label_name, '_GT');
        input_name = [temp{1}, '.png'];
        input_file = [subdir, input_name];
        image = imread(input_file,'png');
        image = image(:,:,1);
        image = single(image) / 255;
        
        [h, w] = size(image);
        
        label_file = [subdir, label_name];
        image_label = imread(label_file,'png');
        image_label = image_label(:,:,1);
        image_label = single(image_label) / 255;
        
        judge = sum(sum(image_label));
        disp(input_name);
        disp(label_name);
        disp(['number of whistles : ', num2str(judge)])
        if sum(sum(image_label)) == 0
            continue
        end
        %         figure(1);
        %         imshow(image);
        %         figure(2);
        %         imshow(image_label);
        [h_l, w_l] = size(image_label);
        h = min(h, h_l);
        w = min(w, w_l);
        for x = 1 : stride : h - patchsize + 1
            for y = 1 : stride : w - patchsize + 1
                data(:,:,1,count + 1) = image(x:x+patchsize-1, y:y+patchsize-1);
                label(:,:,1,count + 1) = image_label(x:x+patchsize-1, y:y+patchsize-1);
                count = count + 1;
            end
        end
        disp(['number of data : ', num2str(count)])
    end
end

order = randperm(count);
if count > 5000
    order = order(1:5000);
end
data = data(:,:,:, order);
label = label(:,:,:,order);
savepath = [output_dir, '/test_', data_group,'_patch', num2str(patchsize),'_stride', num2str(stride), '.h5'];
chunksz = 128;
created_flag = false;
totalct = 0;
for batchno = 1: floor(size(data, 4) / chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:, :, :, last_read+1:last_read+chunksz);
    batchlabs = label(:, :, :, last_read+1:last_read+chunksz);
    
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);