clc;
clear all;

transcript_dir='/export/b03/xwang/mixer6/kaldi/egs/rm/s5/data/local/annotations';
transflist=[transcript_dir,'/trans.flist'];
outfile=[transcript_dir,'/train.txt'];
fid_out=fopen(outfile,'w+');
fid=fopen(transflist);
line=fgetl(fid);

while (ischar(line))
    idx=strfind(line,'/');
    trans_dir=['/export/b03/xwang/mixer6/kaldi/egs/rm/s5/',line];
    trans_name=line(idx(length(idx))+1:end);
    % get the speaker id and date and 
    idx2=strfind(trans_name,'_');
    spk_id=trans_name(1:idx2(1)-1);
    date=trans_name(idx2(1)+1:idx2(2)-1);
    digit_pin=trans_name(idx2(2)+1:idx2(3)-1);
%    display(trans_dir);
    fid_trans=fopen(trans_dir);
    line_trans=fgetl(fid_trans);
 %   display(line_trans);
    % process one-by-one
    while (ischar(line_trans))
        idx3=strfind(line_trans,',');
        start_point=line_trans(1:idx3(1)-1);
        end_point=line_trans(idx3(1)+1:idx3(2)-1);
        word=line_trans(idx3(2)+1:end);
        outline=[digit_pin,' ',date,' ',spk_id,' ',start_point,' ',end_point,' ',word];        
        fprintf(fid_out,outline); 
        fprintf(fid_out,'\n');
        line_trans=fgetl(fid_trans);
    end
    fclose(fid_trans);
    line=fgetl(fid);
end

fclose(fid);
fclose(fid_out);
