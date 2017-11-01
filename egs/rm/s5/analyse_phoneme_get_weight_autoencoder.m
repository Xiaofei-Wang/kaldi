function  weight = analyse_phoneme_get_weight_autoencoder(mData)
% Analyse 
num_stream=length(mData);
[phone_num, frame_num] = size(mData{1});
autoencoder_vec=zeros(num_stream, frame_num);
weight=zeros(num_stream,frame_num);

%    for j = 1 : frame_num
        for p = 1 : num_stream
            tmp=mData{p}; % got vector of each frame
            autoencoder_vec(p,:)=1./(tmp.^2);
        end
        for p = 1 : num_stream
            weight(p,:)=autoencoder_vec(p,:)./sum(autoencoder_vec);
        end
%    end

end