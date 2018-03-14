function combined_post=filtposterior_use_weight(mData, weight, prior_prob)
%function combined_post=filtposterior(mData)
num_stream=length(mData);
[state_num, frame_num] = size(mData{1});
state_vec=zeros(num_stream, state_num);
weight_vec=zeros(num_stream, 1);
% check the size of mData
for p = 2 : num_stream
   if length(mData{p}(1,:)) < frame_num
	   mData{p}(:,length(mData{p}(1,:))+1:frame_num)=zeros(state_num,frame_num-length(mData{p}(1,:)));
   end
end


combined_post=zeros(state_num,frame_num);
    for j = 1 : frame_num
        for p = 1 : num_stream
            state_vec(p,:)=mData{p}(:,j); % got vector of each frame            
        end
        weight_vec=weight(:,j);
        combined_post(:,j)=state_vec'*weight_vec;
        combined_post(:,j)=log(combined_post(:,j))-log(prior_prob);
    end
end
