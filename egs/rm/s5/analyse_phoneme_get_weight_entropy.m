function  weight = analyse_phoneme_get_weight_entropy(mData)
% Analyse 
num_stream=length(mData);
[phone_num, frame_num] = size(mData{1});
entropy_vec=zeros(num_stream, frame_num);
weight=zeros(num_stream,frame_num);

%    for j = 1 : frame_num
        for p = 1 : num_stream
            tmp=mData{p}; % got vector of each frame
            entropy_vec(p,:)=1./calc_entropy(tmp);
        end
        for p = 1 : num_stream
            weight(p,:)=entropy_vec(p,:)./sum(entropy_vec);
        end
%    end

end