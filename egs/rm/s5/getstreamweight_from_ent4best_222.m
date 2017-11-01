clc;
clear all;
% This script is used for generate the combination of multi-streams

phone_posterior_dir='data-posterior/';
weight_dir='exp/dnn4_pretrain-dbn_dnn/decode_';
num_streams = 13;

tasks={'dev','test'};
scenes={'CH01','CH02','CH03','CH04','CH05','CH06','CH07','CH08','CH09','CH10','CH11','CH12','CH13'};


% got the features
for i1 = 1: length(tasks)
    task=tasks{i1};
%    for i2 = 1: length(scenes)
        
%    scene=scenes{i2};
    
    posterir_name=[phone_posterior_dir,'posterior_',task,'.mat'];
    load(posterir_name);

    number_utt=length(features{1}.utt);
    out_features=struct('utt',cell(1),'feature',cell(1));

    % process each utterance frame by frame
    for utt_id = 1 : number_utt
        % got the utterance matrix with number of channels
        utt_name=features{1}.utt{utt_id};
        tmp{1} = features{4}.feature{utt_id};
        tmp{2} = features{6}.feature{utt_id};
        tmp{3} = features{10}.feature{utt_id};
        tmp{4} = features{11}.feature{utt_id};
    
        enhanced_phoneme = analyse_phoneme_get_weight_entropy(tmp);
    
        out_features.utt{utt_id}=utt_name;
        out_features.feature{utt_id}=enhanced_phoneme;
    
        display(['sentence ',num2str(utt_id),' (out of ',num2str(number_utt),') finished!']);    
    end
    outdir=[weight_dir,task,'_ent4best_comb/weight'];
    mkdir(outdir);
    outfilename=[outdir,'/stream_weights.ark'];
    writekaldifeatures(out_features, outfilename);
    
%    end
end



            



