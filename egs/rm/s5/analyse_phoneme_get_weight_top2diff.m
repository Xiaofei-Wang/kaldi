function  weight = analyse_phoneme_get_weight_top2diff(mData)
% Analyse 
num_stream=length(mData);
[phone_num, frame_num] = size(mData{1});
top2diff_vec=zeros(num_stream, frame_num);
weight=zeros(num_stream, frame_num);

    for j = 1 : frame_num
        for p = 1 : num_stream
            tmp=mData{p}(:,j); % got vector of each frame
            tmp2=sort(tmp,'descend');
            top2diff_vec(p,j)=tmp2(1)-tmp2(2);          
        end
    end
    for p = 1 : num_stream
        weight(p,:)=top2diff_vec(p,:)./sum(top2diff_vec);
    end

end