#!/bin/bash
for JOB in $( seq 1 100 ); do
echo "

% This script is used for generate the combination of multi-streams
% CH02 are not use for combination

weight_dir='exp/dnn4_pretrain-dbn_dnn/decode_';
num_streams = 13;

tasks={'dev'};
scenes={'CH01','CH02','CH03','CH04','CH05','CH06','CH07','CH08','CH09','CH10','CH11','CH12','CH13'};

prior_prob=readpriorfile('exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts');

% got the features
for i1 = 1: length(tasks)

  task=tasks{i1};
			        
  % process the utterance job-by-job
  for job=$JOB 
    out_features=struct('utt',cell(1),'feature',cell(1));

    % read the stage posterior file
    for i2 = 1: length(scenes)

      scene=scenes{i2};
      posterior_file=['exp/dnn4_pretrain-dbn_dnn/decode_',task,'_',scene,'_post/post/post.',num2str(job),'.scp'];
      state_posterior{i2}=readkaldifeatures(posterior_file);

    end

    ark_utt_num=length(state_posterior{1}.utt);
    for j = 1 : ark_utt_num
	    utt_name=state_posterior{1}.utt{j};
	    feature{1}=state_posterior{4}.feature{j};
	    feature{2}=state_posterior{6}.feature{j};
	    feature{3}=state_posterior{10}.feature{j};
	    feature{4}=state_posterior{11}.feature{j};

	    % Process the state-posterior
	    out_features.utt{j}=utt_name;

	    out_features.feature{j}=filtposterior(feature, prior_prob);       
	    display(['sentence ',num2str(j),' (out of ',num2str(ark_utt_num),') finished!']);
    end
    outdir=['exp/dnn4_pretrain-dbn_dnn/decode_dev_subsum_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn4_pretrain-dbn_dnn/decode_dev_subsum_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);   
  end

  end

" > local/pc/subsum/postcombination_subsum_${JOB}.m

done
