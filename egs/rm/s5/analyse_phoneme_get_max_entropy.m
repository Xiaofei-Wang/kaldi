function  weight = analyse_phoneme_get_max_entropy(mData)
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
	[V,I]=max(entropy_vec);
        for p = 1 : frame_num
            weight(I(p),p)=1;
        end
%    end

end
