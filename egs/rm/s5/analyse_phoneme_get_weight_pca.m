function  weight = analyse_phoneme_get_weight_pca(mData)
% Analyse 
num_stream=length(mData);
[phone_num, frame_num] = size(mData{1});
phoneme_vec=zeros(num_stream, phone_num);
weight=zeros(num_stream,frame_num);

%for i = 1 : phone_num
    for j = 1 : frame_num
        for p = 1 : num_stream
            phoneme_vec(p,:)=mData{p}(:,j); % got vector of each frame
        end
%       corr_self=phoneme_vec*phoneme_vec';
       coeff=pca(phoneme_vec');
        weight(:,j)=coeff(:,1);
        weight(:,j)=weight(:,j)./sum(weight(:,j));
%       [num,val]=sort(pca_trans);
%       pca_trans(val(1:phone_num-k))=0;
%       pca_trans=pca_trans./sum(pca_trans);
%       output(:,j)=pca_trans;

    end
%end
end