function combined_post=posterior2likelihood(mData, prior_prob)

[state_num, frame_num] = size(mData);

combined_post=zeros(state_num,frame_num);
    for j = 1 : frame_num
        combined_post(:,j)=log(mData(:,j))-log(prior_prob);
    end
end