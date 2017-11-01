clc;
clear all;
% This script is used for generate posterior .mat file from data-posterior

phone_posterior_dir='data-posterior/';
num_streams = 13;

tasks={'dev','test'};
scenes={'CH01','CH02','CH03','CH04','CH05','CH06','CH07','CH08','CH09','CH10','CH11','CH12','CH13'};

% got the features
for i1 = 1: length(tasks)
    task=tasks{i1};
    for i2 = 1: length(scenes)
         scene=scenes{i2};
         % read the posterior features
         feature_name=[phone_posterior_dir,task,'/',scene,'/feats.scp'];
         display(i2);
         features{i2}=readkaldifeatures(feature_name);
    end
    outname=[phone_posterior_dir,'/posterior_',task];
    save(outname,'features','-v7.3');
end
  


            



