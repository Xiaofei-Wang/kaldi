#!/bin/bash

mkdir -p local/pc/energy_beam

for JOB in $( seq 1 100 ); do
echo "

% This script is used for generate the combination of multi-streams

source_dir='data/';
weight_dir='exp/dnn_pretrain-dbn_dnn/decode_';
%num_streams = 13;

tasks={'sim','real'};

flag=0;
prior_prob=readpriorfile('exp/dnn_pretrain-dbn_dnn/ali_train_pdf.counts');

%mics={'LA6'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6'};
mics={'L1C','L4L','L3L','L2R','Beam_Circular_Array','Beam_Linear_Array'};
num_streams=length(mics);

% got the features
for i1 = 1: length(tasks)
   
      task=tasks{i1};
         
      % read the wav.scp files
      for i2 = 1: length(mics)
        mic=mics{i2};
        wavlist_file=[source_dir,'dirha_',task,'_',mic,'/wav.scp'];
	    display(wavlist_file);
        [names{i2}, wavfile{i2}] = textread(wavlist_file,'%s %s ');      
      end

    number_utt=length(names{1});

 % process the utterance job-by-job
  for job=$JOB
    
    out_features=struct('utt',cell(1),'feature',cell(1));
    
    % read the stage posterior file
      mic=mics{1};
      posterior_file=['exp/dnn_pretrain-dbn_dnn/decode_dirha_',task,'_',mic,'_post/post/post.',num2str(job),'.scp'];
      state_posterior=readkaldifeatures(posterior_file);

     
    ark_utt_num=length(state_posterior.utt);
    
    for j = 1 : ark_utt_num
        utt_name=state_posterior.utt{j};

        out_features.utt{j}=utt_name;
        display(utt_name);
        
        % obtain the weight for such utterance
        buffer = zeros(num_streams,1);
        
        for i2 = 1 : num_streams
            for pp = 1: number_utt
                tmp_name=names{i2}(pp);
                flag = strcmp(utt_name, tmp_name);

                if flag
                    [x, fs] = audioread(wavfile{i2}{pp});
                    buffer(i2) = sum(x.^2)/length(x);
                    flag = 0;
                end

            end            
        end
        
        display(buffer);
        [ma, best_stream_idx] = max(buffer);
        display(best_stream_idx);
        posterior_file=['exp/dnn_pretrain-dbn_dnn/decode_dirha_',task,'_',mics{best_stream_idx},'_post/post/post.',num2str(job),'.scp'];

	display(posterior_file);
        state_posterior2=readkaldifeatures(posterior_file);
        
        out_features.feature{j}=posterior2likelihood(state_posterior2.feature{j}, prior_prob);  

	display(['sentence ',num2str(j),' (out of ',num2str(ark_utt_num),') finished!']);
    % process each utterance frame by frame
    end

    outdir=['exp/dnn_pretrain-dbn_dnn/decode_',task,'_energy_beam_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn_pretrain-dbn_dnn/decode_',task,'_energy_beam_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);
    
  end
  
end


" > local/pc/energy_beam/postcombination_${JOB}.m


done
