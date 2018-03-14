function [utt_name, autoencoder] = read_autoencoder_file(sFile)
% Read files generated with Harish's tool 
fid = fopen(sFile);
%fid = fopen('autoencoder_scores.SimData_dt_for_1ch_far_room3_A_2.txt');
tline = fgetl(fid);
line_num = 1;
while ischar(tline)
    utt_list{line_num}=tline;
    utt_list{line_num} = strrep(utt_list{line_num},'[ ','');
    utt_list{line_num} = strrep(utt_list{line_num},']','');
    tline = fgetl(fid);
    line_num = line_num+1;
end

for i = 1 : length(utt_list)
    idx_space = strfind(utt_list{i},' ');
    utt_name{i}=utt_list{i}(1:idx_space(1)-1);
    autoencoder(i)  = str2num(utt_list{i}(idx_space(1)+1:idx_space(2)-1));
end

utt_name = utt_name';
autoencoder = autoencoder';

