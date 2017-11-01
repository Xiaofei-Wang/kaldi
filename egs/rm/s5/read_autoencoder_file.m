function [utt_name, m_measure] = read_autoencoder_file(sFile)
% Read files generated with Harish's tool 
fid = fopen(sFile);
%fid = fopen('mmeasure_scores.SimData_dt_for_1ch_near_room2_A.txt');
tline = fgetl(fid);
line_num = 1;
while ischar(tline)
    utt_list{line_num}=tline;
    utt_list{line_num} = strrep(utt_list{line_num},'[ ','');
    utt_list{line_num} = strrep(utt_list{line_num},']','');
    tline = fgetl(fid);
    line_num = line_num+1;
end
mmeasure_delta = 0;
for i = 1 : length(utt_list)
    idx_space = strfind(utt_list{i},' ');
    utt_name{i}=utt_list{i}(1:idx_space(1)-1);
%    for j = 1 : 15
%        mmeasure_delta = mmeasure_delta + str2num(utt_list{i}(idx_space(j+5)+1:idx_space(j+6)-1));
%    end
    mmeasure_delta = str2num(utt_list{i}(idx_space(1)+1:idx_space(2)-1));
    m_measure(i) = mmeasure_delta;
    mmeasure_delta = 0;
end

utt_name = utt_name';
m_measure = m_measure';


%str = str(idx_space(1)+1:end);
%vMM = str2num(str);  %#ok<ST2NM>
