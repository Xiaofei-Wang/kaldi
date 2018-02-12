

% This script is used for generate the combination of multi-streams

weight_dir='exp/dnn_pretrain-dbn_dnn/decode_';
mics={'LA6','L1C','L4L','LD07','L3L','L2R'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6','Beam_Circular_Array','Beam_Linear_Array'};
scenes={'sim','real'};
num_streams = length(mics);

prior_prob=readpriorfile('exp/dnn_pretrain-dbn_dnn/ali_train_pdf.counts');

% got the features
for i1 = 1: length(scenes)

  scene=scenes{i1};
  weight_file=[weight_dir,scene,'_','frame_AE_nobeam_comb/weight/stream_weights.scp'];

  % read the weight file with names and weights
  weight_feature=readkaldifeatures(weight_file);
  num_utt=length(weight_feature.utt);
			        
  % process the utterance job-by-job
  for job=80 
    out_features=struct('utt',cell(1),'feature',cell(1));

    % read the stage posterior file
    for i2 = 1: num_streams

      mic=mics{i2};
      posterior_file=['exp/dnn_pretrain-dbn_dnn/decode_dirha_',scene,'_',mic,'_post/post/post.',num2str(job),'.scp'];
      state_posterior{i2}=readkaldifeatures(posterior_file);

    end

    ark_utt_num=length(state_posterior{1}.utt);
    for j = 1 : ark_utt_num
	    utt_name=state_posterior{1}.utt{j};
	    utt_name=strrep(utt_name, mics{1}, 'attention');
	    for p = 1 : num_streams
		    feature{p}=state_posterior{p}.feature{j};
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
    outdir=['exp/dnn_pretrain-dbn_dnn/decode_',scene,'_frame_AE_nobeam_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn_pretrain-dbn_dnn/decode_',scene,'_frame_AE_nobeam_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);   
  end

  end


