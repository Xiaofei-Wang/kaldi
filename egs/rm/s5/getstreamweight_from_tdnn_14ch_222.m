clc;
clear all;
% This script is used for generate the combination of multi-streams

autoencoder_dir='data-tdnn-autoencoder-prove/';
weight_dir='exp/dnn4_pretrain-dbn_dnn/decode_';
num_streams = 14;

tasks={'dev','test'};
scenes={'CH01','CH02','CH03','CH04','CH05','CH06','CH07','CH08','CH09','CH10','CH11','CH12','CH13','CH14'};

% got the features
for i1 = 1: length(tasks)
   
      task=tasks{i1};
         
      % read the autoencoder features
      for i2 = 1: length(scenes)
        scene=scenes{i2};
        feature_name=[autoencoder_dir,task,'/',scene,'/tdnn_autoencoder_mse.ark'];
	display(feature_name);
        features{i2}=readkaldifeatures(feature_name);      
      end

    number_utt=length(features{1}.utt);
    out_features=struct('utt',cell(1),'feature',cell(1));

    % process each utterance frame by frame
    for utt_id = 1 : number_utt
        % got the utterance matrix with number of channels
        utt_name=features{1}.utt{utt_id};
        tmp{1}=features{1}.feature{utt_id};
        for  i = 3 : num_streams
            tmp{i-1} = features{i}.feature{utt_id};
        end
    
        enhanced_phoneme = analyse_phoneme_get_weight_autoencoder(tmp);
    
        out_features.utt{utt_id}=utt_name;
        out_features.feature{utt_id}=enhanced_phoneme;
    
        display(['sentence ',num2str(utt_id),' (out of',num2str(number_utt),') finished!']);    
    end
    outdir=[weight_dir,task,'_frame_tdnn_14ch_comb/weight'];
    mkdir(outdir);
    outfilename=[outdir,'/stream_weights.ark'];
    writekaldifeatures(out_features, outfilename);

end




            



