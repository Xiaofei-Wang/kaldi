

% This script is used for generate the combination of multi-streams
weight_dir='exp/dnn4_pretrain-dbn_dnn/decode_';
mics={'LA6','L1C','L4L','LD07','L3L','L2R'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6','Beam_Circular_Array','Beam_Linear_Array'};
num_streams = length(mics);

flag=0;
prior_prob=readpriorfile('exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts');

% got the features

    for i1 = 1: num_streams
	mic=mics{i1};
	autoencoder_file=['data-autoencoder/dirha_',mic,'/autoencoder_scores.txt'];
	[names{i1}, autoencoder_v{i1}] = read_autoencoder_file(autoencoder_file);
    end

    num_utt=length(names{1});
    display('Finishing reading the autoencoder file.');
			        
  % process the utterance job-by-job
  for job=88

    out_features=struct('utt',cell(1),'feature',cell(1));

    % read the stage posterior file
    for i2 = 1: num_streams

      mic=mics{i2};
      posterior_file=['exp/dnn4_pretrain-dbn_dnn/decode_dirha_',mic,'_post/post/post.',num2str(job),'.scp'];
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
	display(utt_name);
	
	% obtain the weight for such utterance
	buffer = zeros(num_streams,1);

	for i2 = 1 : num_streams
	    for pp = 1: num_utt
		tmp_name=names{i2}(pp);
		tmp_name=strrep(tmp_name, mics{i2}, 'attention');
		flag = strcmp(utt_name, tmp_name);
		
	    if flag
		buffer(i2) = autoencoder_v{i2}(pp);
		flag = 0;
	    end

	    end
	end
	display(buffer);
	[ma, best_stream_idx] = max(buffer);
	display(best_stream_idx);
	out_features.feature{j}=filtposterior(feature{best_stream_idx}, prior_prob);           
	display(['sentence ',num2str(j),' (out of ',num2str(ark_utt_num),') finished!']);
    end


    outdir=['exp/dnn4_pretrain-dbn_dnn/decode_autoencoder_nobeam_comb/post'];
    mkdir(outdir);
    outfilename=['exp/dnn4_pretrain-dbn_dnn/decode_autoencoder_nobeam_comb/post/post.',num2str(job),'.txt'];
    writekaldifeatures(out_features, outfilename);   

  end


