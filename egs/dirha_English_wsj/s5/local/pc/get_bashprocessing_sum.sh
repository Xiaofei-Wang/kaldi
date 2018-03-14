#!/bin/bash

task=$1
mkdir -p local/pc/$task

for JOB in $( seq 1 100 ); do
echo "

% This script is used for generate the combination of multi-streams

weight_dir='exp/dnn_pretrain-dbn_dnn/decode_';
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6'};
mics={'L1C','L4L','LD07','L3L','L2R','Beam_Circular_Array'};
num_streams = length(mics);
scenes={'sim','real'};

prior_prob=readpriorfile('exp/dnn_pretrain-dbn_dnn/ali_train_pdf.counts');

% got the features
for i1 = 1: length(scenes)
  
  scene=scenes{i1};
			        
  % process the utterance job-by-job
  for job=$JOB 
    out_features=struct('utt',cell(1),'feature',cell(1));

    % read the stage posterior file
    for i2 = 1: length(mics)

      mic=mics{i2};
      posterior_file=['exp/dnn_pretrain-dbn_dnn/decode_dirha_',scene,'_',mic,'_post/post/post.',num2str(job),'.scp'];
      state_posterior{i2}=readkaldifeatures(posterior_file);

    end

    ark_utt_num=length(state_posterior{1}.utt);
    for j = 1 : ark_utt_num
	    utt_name=state_posterior{1}.utt{j};
	    for p = 1 : num_streams
		    feature{p}=state_posterior{p}.feature{j};
	    end

	    % Process the state-posterior
	    out_features.utt{j}=utt_name;

	    out_features.feature{j}=filtposterior(feature, prior_prob);       
	    display(['sentence ',num2str(j),' (out of ',num2str(ark_utt_num),') finished!']);
    end
    
    outdir=['exp/dnn_pretrain-dbn_dnn/decode_',scene,'_${task}_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn_pretrain-dbn_dnn/decode_',scene,'_${task}_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);   
  end

  end

" > local/pc/${task}/postcombination_${JOB}.m

done
