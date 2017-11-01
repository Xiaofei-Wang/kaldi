clc;
clear all;

    num_stream = 13;
 %   index_fid=fopen('ppp.txt','w');

    for j = 1 : num_stream
        if j < 10
            autoencoder_name{j} = ['data-autoencoder/test/CH0',num2str(j),'/autoencoder_scores.txt' ];
            score_name{j} = ['exp/dnn4_pretrain-dbn_dnn/decode_test_CH0',num2str(j),'/scoring/1.tra'];
        else   
            autoencoder_name{j} = ['data-autoencoder/test/CH',num2str(j),'/autoencoder_scores.txt' ];
            score_name{j} = ['exp/dnn4_pretrain-dbn_dnn/decode_test_CH',num2str(j),'/scoring/1.tra'];
        end
    end
    % get the best score file
    %first, get the utt list.
    fid = fopen(score_name{1});
%    fid = fopen('15.tra');
    tline = fgetl(fid);
    line_num = 1;
    while ischar(tline)
        file_list{line_num}=tline(1:44);
        tline = fgetl(fid);
        line_num = line_num+1;
    end
    fclose(fid);
    
    for j = 1 : num_stream
%        fid = fopen('15.tra');
       fid = fopen(score_name{j});
        tline = fgetl(fid);
        line_num = 1;
        while ischar(tline)
            score_file{j,line_num}=tline;
            tline = fgetl(fid);
            line_num = line_num+1;
        end
        fclose(fid);
    end
    
%    names = cell(length(file_list),num_stream);
%    autoencoder_v = zeros(length(file_list), num_stream);
    
    %second, compare the autoencoder
    for j = 1 : num_stream
%        [names{j}, autoencoder_v{j}] = textread('autoencoder_scores.SimData_dt_for_1ch_far_room1_A.txt','%s %f ');
        [names{j}, autoencoder_v{j}] = read_autoencoder_file(autoencoder_name{j});
    end
    
    buffer = zeros(num_stream,1);
    for m = 1 : length(file_list)
        pp = file_list(m);
        for j = 1 : num_stream
            for i = 1 : length(file_list)
                flag = strcmp(pp, names{j}(i));
                if flag
                    buffer(j) = autoencoder_v{j}(i);
                end
            end
        end
        buffer(2)=-1;
        [ma, best_stream_idx] = max(buffer);
%        fprintf(index_fid,num2str(best_stream_idx));
%        fprintf(index_fid,'\n');
        best_score{m} = score_file{best_stream_idx, m};         
    end
    
    ofdir = 'exp/dnn4_pretrain-dbn_dnn/decode_test_autoencoder_comb/scoring';
    mkdir(ofdir);
    ofname = [ofdir,'/1.tra'];
    
    out_file = fopen(ofname,'w');
    for m = 1 : length(file_list)
        fprintf(out_file,best_score{m});
        fprintf(out_file,'\n');
    end
    fclose(out_file);
%    fclose(index_fid);
    clear names autoencoder_v best_score file_list score_file


