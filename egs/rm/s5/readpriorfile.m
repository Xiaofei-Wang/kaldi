function prior_prob=readpriorfile(file_name)
    fid=fopen(file_name,'r');
%    fid = fopen('ali_train_pdf.counts','r');
    str = fgetl(fid);
    str = strrep( str, ' [', '' );
    str = strrep( str, ']', '' );
    idx_space = strfind(str,' ');
    prior_prob=zeros(length(idx_space)-1,1);
    for i = 1 : length(idx_space)-1
        tmp=str2num(str(idx_space(i)+1:idx_space(i+1)-1));
        prior_prob(i)=tmp;
    end

    prior_prob=prior_prob./sum(prior_prob);

end