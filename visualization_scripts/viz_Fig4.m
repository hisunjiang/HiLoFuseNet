clear
clc
close all

% 'loc' and 'ctmr' toolboxes are provided by the authors of Stanford fingerflex
% dataset
addpath( genpath('E:\My projects\finger ECoG\codeDemo\loc') ) 
addpath( genpath('E:\My projects\finger ECoG\codeDemo\ctmr') )

% Define experimental metadata
allSubjects = {'bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'};
numSubj = numel(allSubjects);
numPatterns = 20; % Number of spatial filters per group

% Select the Region of Interest (ROI) for spatial specificity analysis
% Options: 'dorsalM1S1' (Hand-knob area), 'Dorsal M1', 'dorsalS1'
ROI = 'dorsalM1'; 
if strcmp(ROI, 'dorsalM1S1')
    ROI_IDX = [1, 3];
    fig_title = 'Dorsal M1+S1';
elseif strcmp(ROI, 'dorsalM1')
    ROI_IDX = [1];
    fig_title = 'Dorsal M1';
elseif strcmp(ROI, 'dorsalS1')
    ROI_IDX = [3];
    fig_title = 'Dorsal S1';
end

hga_roi_ratio = NaN(1, numSubj);
lfs_roi_ratio = NaN(1, numSubj);
valid_subj_mask = false(1, numSubj); % Boolean mask to track subjects containing the target ROI

for iS = 1:numSubj
    subject = allSubjects{iS};
    
    % Load electrode localization coordinates and anatomical region labels
    load(['E:\My projects\finger ECoG\data\Stanford\', subject, '\', subject, '_fingerflex.mat'], 'locs', 'elec_regions');
    % Load spatial filter kernels for both High-Gamma Activity (HGA) and Low-Frequency Signals (LFS)
    load(['E:\My projects\finger ECoG\code\HiLoFuseNet\finger_regression\results\o5\interpretModel\spatial_kernels_', subject, '.mat']);
    
    % Identify valid, clean channels
    total_chans = size(locs, 1);
    channelName = arrayfun(@(x) sprintf('ch%d', x), 1:total_chans, 'UniformOutput', false)';

    badChans = {};
    if strcmp(subject, 'jp'), badChans = {'ch31','ch32','ch37'};
    elseif strcmp(subject, 'mv'), badChans = {'ch40'}; end
   
    isBad = ismember(channelName, badChans);
    good_indices = find(~isBad); 
    electrodes = locs(good_indices, :);
    elec_regions_clean = elec_regions(good_indices);
            
    % Locate the column indices corresponding to the target ROI
    m1s1_idx = find(ismember(elec_regions_clean, ROI_IDX));
   
    % CRITICAL SAFETY CHECK: If the subject doesn't have any electrodes implanted 
    % in the target ROI, exclude them from the statistical comparison to prevent artifacts.
    if isempty(m1s1_idx)
        fprintf('Warning: Subject [%s] has NO electrodes in %s. Skipped from analysis.\n', subject, fig_title);
        continue; 
    end
    
    % Mark this subject as valid for the final group-level calculation
    valid_subj_mask(iS) = true;

    [num_filters, num_channels] = size(kernels_group1);
    HGA_weight_normalized = zeros(numPatterns, num_channels);
    LFS_weight_normalized = zeros(numPatterns, num_channels);
    
    for i = 1:numPatterns
        % Normalize HGA spatial weights: square to eliminate phase/polarization cancellation, 
        % then rescale [0, 1] to prevent amplitude scaling domination.
        squared_weights = kernels_group1(i,:).^2;
        min_val = min(squared_weights); max_val = max(squared_weights);
        if max_val > min_val
            HGA_weight_normalized(i, :) = (squared_weights - min_val) / (max_val - min_val);
        else
            HGA_weight_normalized(i, :) = zeros(size(squared_weights));
        end
        
        % Normalize LFS spatial weights using the identical mathematical standard
        squared_weights = kernels_group2(i,:).^2;
        min_val = min(squared_weights); max_val = max(squared_weights);
        if max_val > min_val
            LFS_weight_normalized(i, :) = (squared_weights - min_val) / (max_val - min_val);
        else
            LFS_weight_normalized(i, :) = zeros(size(squared_weights));
        end
    end
    
    % Extract the grand average spatial attention profile across all 20 patterns
    w_hga = mean(HGA_weight_normalized, 1);
    w_lfs = mean(LFS_weight_normalized, 1);
    
    % Calculate what proportion of the total spatial attention model energy is concentrated inside the ROI
    hga_roi_ratio(iS) = sum(w_hga(m1s1_idx)) / sum(w_hga);
    lfs_roi_ratio(iS) = sum(w_lfs(m1s1_idx)) / sum(w_lfs);
end

% Filter out the NaN values from skipped subjects to prepare clean vectors
final_hga_ratio = hga_roi_ratio(valid_subj_mask)';
final_lfs_ratio = lfs_roi_ratio(valid_subj_mask)';
num_valid_subjs = sum(valid_subj_mask);

% Perform a non-parametric Two-tailed Paired Wilcoxon Signed-Rank Test 
% (Robust against non-normal distributions and smaller sample sizes, N < 30)
[p_ratio, ~, stats1] = signrank(final_hga_ratio, final_lfs_ratio);

fprintf('--- Statistical Summary (N = %d/%d valid subjects) ---\n', num_valid_subjs, numSubj);
fprintf('Spatial Focus Fraction: HGA(%.2f%%) vs LFS(%.2f%%) | p-value = %.4f\n', ...
    mean(final_hga_ratio)*100, mean(final_lfs_ratio)*100, p_ratio);

fig1 = figure;
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [5, 5, 3, 4.]);
hold on;

ratio_data_hga = final_hga_ratio * 100;
ratio_data_lfs = final_lfs_ratio * 100;

% Plot Paired Connecting Lines
for iP = 1:num_valid_subjs
    plot([1, 2], [ratio_data_hga(iP), ratio_data_lfs(iP)], '-', ...
        'Color', [0.5 0.5 0.5 0.3], 'LineWidth', 0.8); 
end

% Plot Jittered Scatter Data Points
scatter(ones(size(ratio_data_hga)) + (rand(size(ratio_data_hga))-0.5)*0.15, ...
    ratio_data_hga, 15, [0.8 0.2 0.2], 'filled', 'MarkerFaceAlpha', 0.6);
scatter(2*ones(size(ratio_data_lfs)) + (rand(size(ratio_data_lfs))-0.5)*0.15, ...
    ratio_data_lfs, 15, [0.2 0.2 0.8], 'filled', 'MarkerFaceAlpha', 0.6);

% Superimpose Distribution Boxplots
ratio_data = [ratio_data_hga, ratio_data_lfs]; 
box_h1 = boxplot(ratio_data, 'Labels', {'HGA', 'LFS'}, 'Widths', 0.4, 'Colors','k');
set(box_h1, 'LineWidth', 1.0);

% Label and aesthetic configurations
ylabel('Spatial Focus Fraction (%)','FontSize',6); 
box off;
set(gca, 'FontSize', 6, 'TickDir', 'out');

yl = ylim;
y_line = yl(2) + (yl(2)-yl(1))*0.05;
y_text = yl(2) + (yl(2)-yl(1))*0.08;
ylim([yl(1), yl(2) + (yl(2)-yl(1))*0.15]); 

% Draw significance bar
plot([1, 2], [y_line, y_line], '-k', 'LineWidth', 0.8);

% Convert numerical p-value into classic academic significance stars
if p_ratio < 0.001, sig_str = '***'; 
elseif p_ratio < 0.01, sig_str = '**'; 
elseif p_ratio < 0.05, sig_str = '*'; 
else sig_str = 'n.s.'; end

if strcmp(sig_str, 'n.s.')
    text(1.5, y_text+0.3, sig_str, 'FontSize', 8, 'HorizontalAlignment', 'center');
else
    text(1.5, y_text, sig_str, 'FontSize', 12, 'HorizontalAlignment', 'center');
end

title(fig_title,'FontSize', 8, 'FontWeight', 'bold' )

print(fig1, ['E:\my papers\finger decoding\HiLoFuseNet\figures\weight_ratio_', ROI, '.tiff'], '-dtiff', '-r300');
