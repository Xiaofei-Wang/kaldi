clc;
clear all;
chn=14;
sen_num=0;

wav_dir='/export/corpora5/LDC/LDC2013S03/mx6_speech/data/pcm_flac/';
out_dir='/export/b03/xwang/data/mixer6/test/';
fid=fopen('iv_components_final2.csv');
line=fgetl(fid);

while (ischar(line))
    line=fgetl(fid);
    sen_num=sen_num+1;
    idx=strfind(line,',');
    name=line(1:idx(1)-1);
    rdtr_bgn=line(idx(5)+1:idx(6)-1);
    rdtr_end=line(idx(6)+1:idx(7)-1);
    
    for i = 1 : 14
        if i < 10
            wavname=[wav_dir,'/CH0',num2str(i),'/',name,'_CH0',num2str(i),'.flac'];
        else
            wavname=[wav_dir,'/CH',num2str(i),'/',name,'_CH',num2str(i),'.flac'];
        end
%        wavname='20090713_091520_LDC_120304_CH01.flac';
        % check if the target file exist
        if exist(wavname, 'file')
            % File exists.  Do stuff....
            [xx,fs]=audioread(wavname);
            pp = xx(fs*str2num(rdtr_bgn):fs*str2num(rdtr_end));
            if i < 10
                outdir2=[out_dir,'/CH0',num2str(i)];
                wavname2=[outdir2,'/',name,'_CH0',num2str(i),'.wav'];
            else
                outdir2=[out_dir,'/CH',num2str(i)];
                wavname2=[outdir2,'/',name,'_CH',num2str(i),'.wav'];
            end
            mkdir(outdir2);
            audiowrite(wavname2,pp,fs);
        else
            % File does not exist.
            warningMessage = sprintf('Warning: file does not exist:\n%s', wavname);
            display(warningMessage);
        end
    end
end
fclose(fid);

