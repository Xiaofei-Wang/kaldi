clc;
clear all;
% This script is used for generate the combination of multi-streams

phone_posterior_dir='data-posterior/';
weight_dir='exp/dnn_pretrain-dbn_dnn/decode_';

tasks={'sim','real'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6'};
mics={'L1C','L4L','LD07','L3L','L2R','Beam_Circular_Array'};
num_streams = length(mics);
for i1 = 1 : length(tasks)

    task = tasks{i1};
    % got the features
    
    posterir_name=[phone_posterior_dir,'posterior_',task,'_',num2str(num_streams),'mics'];
    load(posterir_name);

    number_utt=length(features{1}.utt);
    out_features=struct('utt',cell(1),'feature',cell(1));

    % process each utterance frame by frame
    for utt_id = 1 : number_utt
        % got the utterance matrix with number of channels
        utt_name=features{1}.utt{utt_id};
        utt_name=strrep(utt_name, mics{1}, 'attention');
        for  i = 1 : num_streams
            tmp{i} = features{i}.feature{utt_id};
        end
    
        enhanced_phoneme = analyse_phoneme_get_weight_entropy(tmp);
    
        out_features.utt{utt_id}=utt_name;
        out_features.feature{utt_id}=enhanced_phoneme;
    
        display(['sentence ',num2str(utt_id),' (out of ',num2str(number_utt),') finished!']);    
    end
    outdir=[weight_dir,task,'_frame_entropy_1beam_comb/weight'];
    mkdir(outdir);
    outfilename=[outdir,'/stream_weights.ark'];
    writekaldifeatures(out_features, outfilename);
    
end


            



