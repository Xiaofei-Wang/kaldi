function  weight = analyse_phoneme_get_nbest_autoencoder(mData,nbest)
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
    
for j = 1 : frame_num
    
    [value,index]=sort(autoencoder_vec(:,j),'descend');
    
    panda = sum(value(1:nbest));
%    for i = 1 : nbest
%        panda = panda + value(i);
%    end
    for i = 1 : nbest
        weight(index(i),j)=value(i)/panda;
    end
end
%      end

end
