clc;
clear all;
% This script is used for generate the combination of multi-streams

aann_dir='exp/Tdnn-AE-logit-pca-state-post_40D_target_cmvn_5layer_tdnn_11_11/';
input_dir=[aann_dir,'input/dirha_'];
output_dir=[aann_dir,'output/dirha_'];

tasks={'sim','real'};
mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6','Beam_Circular_Array','Beam_Linear_Array'};
%mics={'Beam_Linear_Array','KA6','Beam_Circular_Array'};

%scenes={'CH01'};
% got the features
for i1 = 1: length(tasks)
   
      task=tasks{i1};
         
      % read the autoencoder features
      for i2 = 1: length(mics)
        mic=mics{i2};
        input_scp=[input_dir,task,'_',mic,'/input.scp'];
	display(input_scp);
        output_scp=[output_dir,task,'_',mic,'/output.scp'];
        input_features=readkaldifeatures(input_scp); 
	display('finish reading input.');
        output_features=readkaldifeatures(output_scp); 
        display('finish reading output.');

        number_utt=length(input_features.utt);
        auto_features=struct('utt',cell(1),'feature',cell(1));
         
        for utt_id = 1 : number_utt
            utt_name=input_features.utt{utt_id};
            if (utt_name ~= output_features.utt{utt_id})
                display('Input and output mismatch! Error!');
                break;
            end
            auto_features.utt{utt_id}=utt_name;
            tmp=(input_features.feature{utt_id}-output_features.feature{utt_id}).^2;
            mse=sum(tmp,1)./40;
                        
            auto_features.feature{utt_id}=mse;
	    display(utt_id);
        end
        outdir=['data-tdnn-autoencoder-5layer-tdnn-11-11/dirha_',task,'_',mic];
        mkdir(outdir);
        outfilename=[outdir,'/tdnn_autoencoder_mse.ark'];
        writekaldifeatures(auto_features, outfilename);
       
      end

end




            



