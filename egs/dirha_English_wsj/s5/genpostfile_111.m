clc;
clear all;
% This script is used for generate posterior .mat file from data-posterior

phone_posterior_dir='data-posterior/';

tasks={'sim', 'real'};
%mics={'LA6','L1C','L4L','LD07','L3L','L2R','KA6'};
mics={'L1C','L4L','L3L','L2R','Beam_Circular_Array','Beam_Linear_Array'};

% got the features

num_streams = length(mics);

for i1 = 1: length(tasks)
	task = tasks{i1};
  for i2 = 1: num_streams
     mic=mics{i2};
     % read the posterior features
     feature_name=[phone_posterior_dir,'dirha_',task,'_',mic,'/feats.scp'];
     display(i2);
     features{i2}=readkaldifeatures(feature_name);
  end

  outname=[phone_posterior_dir,'/posterior_',task,'_',num2str(num_streams),'mics'];
  save(outname,'features','-v7.3');

end
