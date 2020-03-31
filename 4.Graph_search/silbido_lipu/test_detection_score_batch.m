clear all; close all;


USE_GRAPH_BASELINE = false; % use graph search method only if true
USE_NET_OUTPUT = true;  % use network output + graph search method if true


test_groups = '2011paper_dc'; % directory to .wav and .bin files. 
det_group = '02-25-2019-4_dc'; % directory to network output


% directory to .wav and .bin files. 
test_group = test_groups;
test_dir = ['/home/lipu/software/silbido/silbido-beta2_lipu/test_data3/', test_group];
if ~exist(test_dir)
    exit();
end
test_files = dir([test_dir, '/*.wav']);

% directory to silbido detection output. 
det_group = det_group;
det_output_dir = ['./detections/', det_group, '/'];

if ~exist(det_output_dir)
    mkdir(det_output_dir);
end

% directory to network output
prop_map_dir = '/home/lipu/software/silbido/silbido-beta2_lipu/prop_maps/';
model_names = dir([prop_map_dir, det_group, '/*']);
model_names = model_names(3:end);

% run detection for each model's output 
for m = 1 : length(model_names)
model_name = model_names(m).name;
prop_maps_dir = ['/home/lipu/software/silbido/silbido-beta2_lipu/prop_maps/', det_group, '/', model_name, '/'];

if USE_NET_OUTPUT
    output_dir = [det_output_dir, '/', model_name];
    if ~exist(output_dir)
        mkdir(output_dir);
    end
end 

if USE_GRAPH_BASELINE
    baseline_name = 'graphs_5k_50k';
    output_dir2 = [det_output_dir, '/', baseline_name];
    if ~exist(output_dir2)
        mkdir(output_dir2);
    end
end

for i = 1 : length(test_files)
    test_file = [test_dir, '/', test_files(i).name];
    temp = strsplit(test_files(i).name, '.');
    test_basename = temp{1};
    
    if USE_NET_OUTPUT
        single_path = [prop_maps_dir, '/', test_basename];
        if ~exist(single_path)
            continue;
        end
        [tonals, subgraphs] = dtTonalsTracking(test_file, 0, Inf, 'Range',...
            [5000, 50000], 'Threshold', 0.5, 'UsePropMap', true, 'ResultPath', single_path);
        dtTonalsSave([output_dir, '/', test_basename,'.det'], tonals);
    end
    
    if USE_GRAPH_BASELINE
        [tonals, subgraphs] = dtTonalsTracking(test_file, 0, Inf, 'Range', [5000, 50000], 'UsePropMap', false);
        dtTonalsSave([output_dir2, '/', test_basename, '.det'], tonals);
    end
end
end


%% Run score script, result in a log file of method performance. 
USE_GRAPH_BASELINE = false;

test_group = test_groups;
test_dir = ['/home/lipu/software/silbido/silbido-beta2_lipu/test_data3/', test_group];
test_files = dir([test_dir, '/*.wav']);

det_group = det_group;
det_output_dir = ['./detections/',det_group,'/'];

gt_criteria = [10, 0.3, 0.15];
ct_str = '';
for i = 1 : length(gt_criteria)
    ct_str = [ct_str, '_', num2str(gt_criteria(i))];
end

temp = dir([det_output_dir, '/*']);
score_dir = ['./scoring/',det_group, ct_str, '/'];
if ~exist(score_dir)
    mkdir(score_dir);
end

for i = 3:length(temp)
model_name = temp(i).name;
prop_maps_dir = ['./prop_maps/', model_name];

output_dir = [det_output_dir, '/', model_name];

which scoreall
result = scoreall('Corpus', test_dir,	...
'Detections', output_dir, ...
'ResultName', [score_dir, '/', model_name],... % , '_JASA2011baseline'
'GroundTruthCriteria', gt_criteria);
dtAnalyzeResults(result, [score_dir, '/', model_name, '_score_2011paper.summary']);

[temp1, files] = processSummary([score_dir, '/', model_name,  '_score_2011paper.summary']);

end
