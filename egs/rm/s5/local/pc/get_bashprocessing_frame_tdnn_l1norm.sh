#!/bin/bash
mkdir -p local/pc/frame_tdnn_l1norm
for JOB in $( seq 1 100 ); do
echo "

% This script is used for generate the combination of multi-streams
% CH02 are not use for combination

weight_dir='exp/dnn4_pretrain-dbn_dnn/decode_';
num_streams = 13;

tasks={'dev','test'};
scenes={'CH01','CH02','CH03','CH04','CH05','CH06','CH07','CH08','CH09','CH10','CH11','CH12','CH13'};

prior_prob=readpriorfile('exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts');

% got the features
for i1 = 1: length(tasks)

  task=tasks{i1};
  weight_file=[weight_dir,task,'_frame_tdnn_l1norm_comb/weight/stream_weights.scp'];

  % read the weight file with names and weights
  weight_feature=readkaldifeatures(weight_file);
  num_utt=length(weight_feature.utt);
			        
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
	    feature{1}=state_posterior{1}.feature{j};
	    for p = 3 : num_streams
		    feature{p-1}=state_posterior{p}.feature{j};
	    end

	    % Process the state-posterior
	    out_features.utt{j}=utt_name;

	    % obtain the weight for such utterance
	    for pp = 1: num_utt
		if strcmp(utt_name, weight_feature.utt{pp})
		   weight=weight_feature.feature{pp};
		   break;
		end
	    end

	    out_features.feature{j}=filtposterior_use_weight(feature, weight, prior_prob);       
	    display(['sentence ',num2str(j),' (out of ',num2str(ark_utt_num),') finished!']);
    end
    outdir=['exp/dnn4_pretrain-dbn_dnn/decode_',task,'_frame_tdnn_l1norm_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn4_pretrain-dbn_dnn/decode_',task,'_frame_tdnn_l1norm_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);   
  end

  end

" > local/pc/frame_tdnn_l1norm/postcombination_use_frame_tdnn_l1norm_${JOB}.m

done
