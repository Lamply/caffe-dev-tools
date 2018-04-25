function [precision, recall] = compute_PR(input_path, seg_path, thresh)
%
% [precision, recall] = COMPUTE_PR(input_path, seg_path, thresh)
% 
%     input_path: input dataset file path
%
%     seg_path: segment result dir path
%
%     thresh: linspace threshold for PR curve compute, [1xn]

%% Before Optimizationload
% fid = fopen(input_path);
% file_list = textscan(fid, '%s %s');
% files_num = size(file_list{1}, 1);
% points = size(thresh, 2);
% 
% precision = zeros(1, points);
% recall = zeros(1, points);
% img_list = file_list{1};
% label_list = file_list{2};
% for i = 1:points
%     tmp_p = zeros(files_num, 1);
%     tmp_r = zeros(files_num, 1);
%     parfor cnt = 1:files_num
%         filename = split(img_list(cnt), '/');
%         filename = char(filename(end));
%         matname = replace(filename, '.png', '.mat');
%         seg_result = load(fullfile(seg_path, matname));
%         prediction = 1.0 * (seg_result.pred > thresh(i));
%         groundtruth = imread(char(label_list(cnt)));
%         groundtruth = double(1.0 - groundtruth);
%         TP = prediction .* groundtruth;
%         tmp_p(cnt) = sum(TP(:))/(sum(prediction(:))+(1e-7));
%         tmp_r(cnt) = sum(TP(:))/(sum(groundtruth(:))+(1e-7));
%     end
%     precision(i) = mean(tmp_p);
%     recall(i) = mean(tmp_r);
% end
% fclose(fid);

%% After Optimization
fid = fopen(input_path);
file_list = textscan(fid, '%s %s');
files_num = size(file_list{1}, 1);
points = size(thresh, 2);

if files_num <= 0
    return;
end
if points <= 0
    return;
end

precision = zeros(1, points);
recall = zeros(1, points);
img_list = file_list{1};
label_list = file_list{2};

groundtruth = {};
seg_result = {};
for cnt = 1:files_num
    groundtruth{cnt} = 1 - imread(char(label_list(cnt)));
    filename = split(img_list(cnt), '/');
    filename = char(filename(end));
    matname = replace(filename, '.png', '.mat');
    seg_result{cnt} = load(fullfile(seg_path, matname));
end

tmp_p = zeros(files_num, points);
tmp_r = zeros(files_num, points);
prediction = zeros(size(groundtruth{1}));
TPFN = zeros(1, files_num);
for cnt = 1:files_num
    TPFN(cnt) = sum(groundtruth{cnt}(:))+(1e-7);
end

for i = 1:points
    for cnt = 1:files_num
        prediction = uint8(seg_result{cnt}.pred > thresh(i));
        TP = prediction .* groundtruth{cnt};
        sTP = sum(TP(:));
        tmp_p(cnt, i) = sTP/(sum(prediction(:))+(1e-7));
        tmp_r(cnt, i) = sTP/TPFN(cnt);
    end
end
precision = mean(tmp_p, 1);
recall = mean(tmp_r, 1);

fclose(fid);

end

