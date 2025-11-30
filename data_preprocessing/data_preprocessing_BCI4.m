%
% DESCRIPTIONS:
%   Preprocessing code for BCI4 datasets. The datasets include 3 subjects,
%   ECoG sampled at 1kHz, band pass filtered between 0.15 to 200 Hz. The
%   data glove was sampled at 25 Hz from the original paper [1], but provided
%   in 1kHz format from the BCI competition [2].
%
% DEPENDENCE:
%   FieldTrip toolbox: fieldtrip-20230926
%   
% REFERENCE:
%   [1] Kubanek, et al., 2009.
%   [2] https://www.bbci.de/competition/iv/#dataset4
% 

close all
clear

fs = 1000;

% locate your data folder
trainDataLoc = './data/BCIIV/BCICIV_4_mat/'; % sub1_comp.mat, sub2_comp.mat, sub3_comp.mat
testLabelLoc = './data/BCIIV/true_labels/'; % sub1_testlabels.mat, sub2_testlabels.mat, sub3_testlabels.mat

% set configurations for data visualization
cfg = [];
cfg.viewmode = 'vertical';  
cfg.blocksize = 10;   
cfg.ylim = [-150, 150];  
cfg.preproc.demean = 'no';
cfg.preproc.lpfilter = 'yes';
cfg.preproc.lpfreq = 100;
cfg.preproc.hpfilter = 'yes';
cfg.preproc.hpfreq = 1;

for iS = 1:3
    load([trainDataLoc, 'sub', num2str(iS),'_comp.mat']);
    load([testLabelLoc, 'sub', num2str(iS),'_testlabels.mat']);
    
    % change the unit
    train_data = train_data'*0.0298;
    test_data = test_data'*0.0298;
    
    % band stop filtering
    train_data = ft_preproc_bandstopfilter(train_data, fs,[59.5 60.5], 3, 'but','onepass');
    train_data = ft_preproc_bandstopfilter(train_data, fs,[119.5, 120.5], 3, 'but','onepass');
    train_data = ft_preproc_bandstopfilter(train_data, fs,[179.5, 180.5], 3, 'but','onepass');
    test_data = ft_preproc_bandstopfilter(test_data, fs,[59.5 60.5], 3, 'but','onepass');
    test_data = ft_preproc_bandstopfilter(test_data, fs,[119.5, 120.5], 3, 'but','onepass');
    test_data = ft_preproc_bandstopfilter(test_data, fs,[179.5, 180.5], 3, 'but','onepass');

    % find and exclude bad channels (interactive)
    nCh = size(train_data, 1);
    channelName = arrayfun(@(x) sprintf('ch%d', x), 1:nCh, 'UniformOutput', false)';
    data.trial = test_data;
    data.label = channelName;
    data.fsample = fs;
    data.time = (1:1:size(data.trial, 2))/data.fsample;

    figure;ft_databrowser(cfg, data);
    visual_badchannel = input('\nEnter faulty channel names: '); % e.g., {'ch1', 'ch2', ...}
    % Below is the removed channels
    % sub3: {'ch23','ch50','ch58'}
    % sub2: {'ch21','ch38'}
    % sub1: {'ch55'}

    idx = find(ismember(data.label, visual_badchannel));

    train_data(idx, :) = [];                       
    test_data(idx, :) = []; 
    channelName(idx) = [];

    % CAR
    train_data = ft_preproc_rereference(train_data, 'all', 'avg');
    test_data = ft_preproc_rereference(test_data, 'all', 'avg');

    % save data
    fprintf('\n subject%d: done!\n', iS)
    save(['./preprocessed_data/BCIIV/sub', num2str(iS),'.mat'], 'train_data', 'test_data', 'train_dg', 'test_dg','fs', 'channelName')
end

