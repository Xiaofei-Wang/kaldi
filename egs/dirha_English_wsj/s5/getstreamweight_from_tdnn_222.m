clc;
clear all;
% This script is used for generate the combination of multi-streams

autoencoder_dir='data-tdnn-autoencoder-5layer-tdnn-11-11/';
weight_dir='exp/dnn_pretrain-dbn_dnn/decode_';
%num_streams = 13;

tasks={'sim','real'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R'};
mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6'};
%mics={'L1C','L4L','L3L','L2R','Beam_Circular_Array','Beam_Linear_Array'};
%mics={'L1C','L4L','LD07','L3L','L2R','Beam_Circular_Array'};

num_streams=length(mics);

% got the features
for i1 = 1: length(tasks)
   
      task=tasks{i1};
         
      % read the autoencoder features
      for i2 = 1: length(mics)
        mic=mics{i2};
        feature_name=[autoencoder_dir,'dirha_',task,'_',mic,'/tdnn_autoencoder_mse.ark'];
	display(feature_name);
        features{i2}=readkaldifeatures(feature_name);      
      end

    number_utt=length(features{1}.utt);
    out_features=struct('utt',cell(1),'feature',cell(1));

    % process each utterance frame by frame
    for utt_id = 1 : number_utt
        % got the utterance matrix with number of channels
        utt_name=features{1}.utt{utt_id};
%        tmp{1}=features{1}.feature{utt_id};
        for  i = 1 : num_streams
            tmp{i} = features{i}.feature{utt_id};
        end
    
        enhanced_phoneme = analyse_phoneme_get_weight_autoencoder(tmp);
    
        out_features.utt{utt_id}=utt_name;
        out_features.feature{utt_id}=enhanced_phoneme;
    
        display(['sentence ',num2str(utt_id),' (out of',num2str(number_utt),') finished!']);    
    end
    outdir=[weight_dir,task,'_frame_tdnn_5layer_tdnn_11_11_nobeam_badmic_comb/weight'];
    mkdir(outdir);
    outfilename=[outdir,'/stream_weights.ark'];
    writekaldifeatures(out_features, outfilename);

end




            



