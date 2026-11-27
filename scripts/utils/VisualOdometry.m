%% Robust Monocular VO: Multi-Set Comparison (6-Set Support)
clear; clc; close all;

% --- 1. Global Parameters ---
%setName = 'c1_ascending_t1_v1'; 
setName = 'c2_sigmoid_t2_v2'; 

KERNEL = 'CAUCHY';

LATENT_DIM = '512D';

LPIPS = 'L_True_256';

GAP = 'GAP_False_0';

TF = 'TF_20';

% FEATURE DETECTION CONFIGURATION
% Options: 'SURF', 'KAZE', 'ORB', 'BRISK'
featureConfig.type = 'KAZE'; 

% --- Added Evaluation Parameters ---
RPE_DELTA = 1;                  % Relative Pose Error interval in frames
KITTI_SEGMENT_LENGTHS = [10 20 30 50 100]; % Segment lengths in frames (frame-based approximation)
MIN_MATCH_COUNT = 30;
MIN_INLIER_COUNT = 20;

% --- Added Robustness Parameters ---
ESSENTIAL_CONFIDENCE = 99.9;
ESSENTIAL_MAX_TRIALS = 5000;
UNDISTORT_OUTPUT_VIEW = 'same';
MASK_BORDER_FRACTION = 0.96;   % Keep central valid area, suppress border artifacts
MIN_VALID_PIXEL_RATIO = 0.25;  % If mask gets too small, fall back to full image

%%
% If this is empty ([]), the Original set is used fully as-is.

%ORIGINAL_EXCLUDE_INDICES = [ ...
%       5   6   9  10  15  16  19  22  24  25  30  33  37  42  45  46  56  60 ...
%      66  68  73  77  79  84  86  90  92  93 108 112 113 124 126 137 139 140 ...
%     146 147 148 154 157 158 159 164 165 179 193 197 207 210 211 213 216 220 ...
%     221 223 226 227 233 238 241 244 247 250 251 259 261 264 268 271 279];

% ORIGINAL_EXCLUDE_INDICES = [108 112 113 124 126 137 139 140];

%c2_sigmoid_t2_v2 | res=336 | latent dim=512 | H=100 | LPIPS=True lpips res=256 | GAN=False | CAUCHY | gap=False size=0 | TF=20%
% End time: 2026-04-18 21:27:58 TUe
ORIGINAL_EXCLUDE_INDICES = [ ...
0   9  11  15  19  22  24  25  30  33  39  42  46  55  56  57  70  72 ...
  73  75  76  77  78  79  82  90  93 101 113 116 117 126 132 137 148 157 ...
 165 173 185 192 196 211 220 229 231 237 238 245 261 275 277 278 284 285 ...
 286 287 290 294 296 298 304 312 314 333 336 346 351 356 360 364 367 369 ...
 373 375 379 392 394 405 406 408 425 427 429 433 438 439 441 442 443 446];


%%
cameraParamsFile = fullfile('Cam_Calib', 'cameraParams.mat');
fprintf('--- Loading Camera Calibration ---\n');

if ~exist(cameraParamsFile, 'file')
    error('Camera calibration file not found: %s', cameraParamsFile);
end

cameraParamsData = load(cameraParamsFile);
cameraParamsFieldNames = fieldnames(cameraParamsData);

if isempty(cameraParamsFieldNames)
    error('cameraParams.mat is empty: %s', cameraParamsFile);
end

cameraParams = [];
for f = 1:numel(cameraParamsFieldNames)
    candidateVar = cameraParamsData.(cameraParamsFieldNames{f});
    if isa(candidateVar, 'fisheyeParameters') || isa(candidateVar, 'fisheyeIntrinsics')
        cameraParams = candidateVar;
        break;
    end
end

if isempty(cameraParams)
    error('No fisheyeParameters or fisheyeIntrinsics object found inside: %s', cameraParamsFile);
end

if isa(cameraParams, 'fisheyeParameters')
    fprintf('Loaded fisheyeParameters object from %s\n', cameraParamsFile);
elseif isa(cameraParams, 'fisheyeIntrinsics')
    fprintf('Loaded fisheyeIntrinsics object from %s\n', cameraParamsFile);
end

% --- 2. Definition of Image Sets ---
sets(1).enabled = true;
sets(1).path    = ['C:\Endo\Images from datasets\C3VDv2\', setName];
sets(1).filter  = ''; 
sets(1).name    = 'Original';

%VAE
sets(2).enabled = true;
sets(2).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(2).filter  = [setName, ' VAE CNNEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(2).name    = 'VAE CC';

sets(3).enabled = false;
sets(3).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(3).filter  = [setName, ' VAE GastroEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(3).name    = 'VAE GC';

sets(4).enabled = false;
sets(4).path    = ['C:\Endo\Recon\', setName, '\Frames']; 
sets(4).filter  = [setName, ' VAE GastroEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(4).name    = 'VAE GE';

sets(5).enabled = false;
sets(5).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(5).filter  = [setName, ' VAE EndoVAEEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(5).name    = 'VAE EE';

%GPVAE_HPA
sets(6).enabled = true;
sets(6).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(6).filter  = [setName, ' GPVAE_HPA CNNEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(6).name    = 'GPVAE_HPA CC';

sets(7).enabled = false;
sets(7).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(7).filter  = [setName, ' GPVAE_HPA GastroEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(7).name    = 'GPVAE_HPA GC';

sets(8).enabled = false;
sets(8).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(8).filter  = [setName, ' GPVAE_HPA GastroEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(8).name    = 'GPVAE_HPA GE';

sets(9).enabled = false;
sets(9).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(9).filter  = [setName, ' GPVAE_HPA EndoVAEEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(9).name    = 'GPVAE_HPA EE';

%GPVAE_SPA
sets(10).enabled = false;
sets(10).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(10).filter  = [setName, ' GPVAE_SPA CNNEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(10).name    = 'GPVAE_SPA CC';

sets(11).enabled = false;
sets(11).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(11).filter  = [setName, ' GPVAE_SPA GastroEnc CNNDecRes ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(11).name    = 'GPVAE_SPA GC';

sets(12).enabled = false;
sets(12).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(12).filter  = [setName, ' GPVAE_SPA GastroEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(12).name    = 'GPVAE_SPA GE';

sets(13).enabled = false;
sets(13).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(13).filter  = [setName, ' GPVAE_SPA EndoVAEEnc EndoVAEDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(13).name    = 'GPVAE_SPA EE';

%MSVQ_VAE
sets(14).enabled = false;
sets(14).path    = ['C:\Endo\Recon\', setName, '\Frames'];
sets(14).filter  = [setName, ' MSVQ_VAE MSVQEnc MSVQDec ',KERNEL,' ',LATENT_DIM,' ',LPIPS,' M_True G_False ',GAP,' ', TF];
sets(14).name    = 'MSVQ_VAE';

setColors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE"]; 

resultsTable = table();

% --- 3. Load Ground Truth ---
gt_filename = ['C:\Endo\C3VDv2\', setName, '\pose.txt'];
fprintf('--- Initializing Diagnostics ---\n');

if exist(gt_filename, 'file')
    gtData = readmatrix(gt_filename);
    gtRaw = gtData(:, 13:15);
    gtCentered = gtRaw - mean(gtRaw);
    fprintf('Ground Truth Loaded: %d poses found for %s.\n', size(gtData, 1), setName);
else
    error('Ground truth file not found: %s', gt_filename);
end

gt_numPoses = size(gtData, 1);
gt_pose_array = zeros(4, 4, gt_numPoses);
for g = 1:gt_numPoses
    if size(gtData, 2) >= 12
        T_gt = eye(4);
        T_gt(1:3,1:4) = reshape(gtData(g, 1:12), [4, 3])';
        gt_pose_array(:,:,g) = T_gt;
    else
        T_gt = eye(4);
        T_gt(1:3,4) = gtCentered(g,:)';
        gt_pose_array(:,:,g) = T_gt;
    end
end

figure('Name', ['VO Comparison: ', setName], 'Color', 'w', 'WindowStyle', 'docked'); 
hold on; grid on; axis equal; view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
title(['Trajectory Comparison (', featureConfig.type, '): ', setName]);
plot3(gtCentered(:,1), gtCentered(:,2), gtCentered(:,3), 'k:', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');

% --- 4. Main Processing Loop ---
for s = 1:numel(sets)
    if ~sets(s).enabled || isempty(sets(s).path), continue; end
    
    currSet = sets(s);
    fprintf('\n>>> Starting Set %d: %s\n', s, currSet.name);
    
    imds = imageDatastore(currSet.path);
    if ~isempty(currSet.filter)
        matchIdx = contains(imds.Files, currSet.filter);
        imds.Files = imds.Files(matchIdx);
    end
    
    if strcmp(currSet.name, 'Original')
        if ~isempty(ORIGINAL_EXCLUDE_INDICES)
            excludeIdx = ORIGINAL_EXCLUDE_INDICES(:);
            excludeIdx = excludeIdx(excludeIdx >= 1 & excludeIdx <= numel(imds.Files));
            keepMask = true(numel(imds.Files), 1);
            keepMask(excludeIdx) = false;
            imds.Files = imds.Files(keepMask);
            fprintf('   Original set: excluded %d indices, kept %d frames.\n', numel(excludeIdx), numel(imds.Files));
        else
            fprintf('   Original set: exclusion list empty, using all frames.\n');
        end
    end
    
    numImages = numel(imds.Files);
    if numImages == 0, continue; end
    
    tempImgRaw = readimage(imds, 1);
    try
        if isa(cameraParams, 'fisheyeParameters')
            [tempImg, intrinsics] = undistortFisheyeImage(tempImgRaw, cameraParams.Intrinsics, 'OutputView', UNDISTORT_OUTPUT_VIEW);
        else
            [tempImg, intrinsics] = undistortFisheyeImage(tempImgRaw, cameraParams, 'OutputView', UNDISTORT_OUTPUT_VIEW);
        end
    catch ME
        error('Failed to undistort first fisheye image: %s', ME.message);
    end
    
    [h, w, ~] = size(tempImg);
    pp = [w/2, h/2]; %#ok<NASGU>
    
    validMask = createValidImageMask(tempImg, MASK_BORDER_FRACTION);
    if mean(validMask(:)) < MIN_VALID_PIXEL_RATIO
        validMask = true(h, w);
    end
    
    vSet = imageviewset;
    vo_pos = zeros(numImages, 3);
    vo_pose_array = repmat(rigidtform3d, numImages, 1);
    is_pose_valid = false(numImages, 1);
    match_count_per_frame = zeros(numImages, 1);
    inlier_count_per_frame = zeros(numImages, 1);
    
    prevI = tempImg;
    prevI = applyImageMask(prevI, validMask);
    if size(prevI, 3) == 3, prevI = rgb2gray(prevI); end
    [prevPoints, prevFeatures] = extractLocalFeatures(prevI, featureConfig.type);
    [prevPoints, prevFeatures] = filterPointsAndFeaturesByMask(prevPoints, prevFeatures, validMask);
    
    vSet = addView(vSet, 1, rigidtform3d(eye(3), [0 0 0]), 'Points', prevPoints);
    vo_pose_array(1) = rigidtform3d(eye(3), [0 0 0]);
    is_pose_valid(1) = true;
    lastViewId = 1; 
    
    for i = 2:numImages
        currIRaw = readimage(imds, i);
        try
            if isa(cameraParams, 'fisheyeParameters')
                [currI, intrinsicsCurr] = undistortFisheyeImage(currIRaw, cameraParams.Intrinsics, 'OutputView', UNDISTORT_OUTPUT_VIEW);
            else
                [currI, intrinsicsCurr] = undistortFisheyeImage(currIRaw, cameraParams, 'OutputView', UNDISTORT_OUTPUT_VIEW);
            end
            intrinsics = intrinsicsCurr;
        catch
            vo_pos(i,:) = vo_pos(i-1,:);
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        currI = applyImageMask(currI, validMask);
        if size(currI, 3) == 3, currI = rgb2gray(currI); end
        
        [currPoints, currFeatures] = extractLocalFeatures(currI, featureConfig.type);
        [currPoints, currFeatures] = filterPointsAndFeaturesByMask(currPoints, currFeatures, validMask);
        
        if isempty(prevFeatures) || isempty(currFeatures)
            vo_pos(i,:) = vo_pos(i-1,:);
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        if size(prevFeatures, 2) ~= size(currFeatures, 2)
             vo_pos(i,:) = vo_pos(i-1,:);
             vo_pose_array(i) = vo_pose_array(i-1);
             is_pose_valid(i) = false;
             continue;
        end
        
        if isa(prevFeatures, 'binaryFeatures') || islogical(prevFeatures) || isa(prevFeatures, 'uint8')
            indexPairs = matchFeatures(prevFeatures, currFeatures, 'Unique', true, 'MatchThreshold', 40);
        else
            indexPairs = matchFeatures(prevFeatures, currFeatures, 'Unique', true, 'MaxRatio', 0.7, 'MatchThreshold', 50);
        end
        
        match_count_per_frame(i) = size(indexPairs, 1);
        
        if size(indexPairs, 1) < MIN_MATCH_COUNT
            vo_pos(i,:) = vo_pos(i-1,:); 
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        matchedPrev = prevPoints(indexPairs(:,1));
        matchedCurr = currPoints(indexPairs(:,2));
        
        try
            [E, inlierIdx, status] = estimateEssentialMatrix( ...
                matchedPrev, matchedCurr, intrinsics, ...
                'Confidence', ESSENTIAL_CONFIDENCE, ...
                'MaxNumTrials', ESSENTIAL_MAX_TRIALS);
        catch
            vo_pos(i,:) = vo_pos(i-1,:);
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        if exist('status', 'var') && ~isempty(status) && status ~= 0
            vo_pos(i,:) = vo_pos(i-1,:);
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        inlier_count_per_frame(i) = sum(inlierIdx);
        
        if sum(inlierIdx) < MIN_INLIER_COUNT
            vo_pos(i,:) = vo_pos(i-1,:);
            vo_pose_array(i) = vo_pose_array(i-1);
            is_pose_valid(i) = false;
            continue;
        end
        
        try
            [relPose, ~] = recoverRelativePose(E, intrinsics, matchedPrev(inlierIdx), matchedCurr(inlierIdx));
        catch
            try
                [R_raw, t_raw, ~] = relativeCameraPose(E, intrinsics, matchedPrev(inlierIdx), matchedCurr(inlierIdx));
                relPose = rigidtform3d(R_raw(:,:,1)', reshape(t_raw(1:3), 1, 3));
            catch
                vo_pos(i,:) = vo_pos(i-1,:);
                vo_pose_array(i) = vo_pose_array(i-1);
                is_pose_valid(i) = false;
                continue;
            end
        end
        
        prevPose = poses(vSet, lastViewId).AbsolutePose;
        currPose = rigidtform3d(prevPose.A * relPose.A);
        
        vSet = addView(vSet, i, currPose, 'Points', currPoints);
        vSet = addConnection(vSet, lastViewId, i, relPose);
        
        vo_pos(i, :) = currPose.Translation;
        vo_pose_array(i) = currPose;
        is_pose_valid(i) = true;
        lastViewId = i; 
        
        prevI = currI;
        prevFeatures = currFeatures;
        prevPoints = currPoints;
        
        if mod(i, 100) == 0
            fprintf('   [Frame %d/%d] Matches: %d | Inliers: %d\n', i, numImages, size(indexPairs, 1), sum(inlierIdx));
        end
    end
    
    minL = min(size(gtCentered, 1), size(vo_pos, 1));
    if minL > 10
        [d, vo_aligned, transform] = procrustes(gtCentered(1:minL, :), vo_pos(1:minL, :), 'Scaling', true, 'Reflection', false); %#ok<ASGLU>
        
        euclidean_errors = sqrt(sum((gtCentered(1:minL, :) - vo_aligned).^2, 2));
        rmse = sqrt(mean(euclidean_errors.^2));
        mean_err = mean(euclidean_errors);
        max_err = max(euclidean_errors);
        median_err = median(euclidean_errors);
        std_err = std(euclidean_errors);
        
        fprintf('   --- %s Stats: Scale %.4f, RMSE %.4f ---\n', currSet.name, transform.b, rmse);
        
        vo_pose_aligned = vo_pose_array(1:minL);
        gt_pose_eval = gt_pose_array(:,:,1:minL);
        
        R_align = transform.T;
        t_align = transform.c(1,:);
        s_align = transform.b;
        
        for k = 1:minL
            T_curr = vo_pose_aligned(k).A;
            R_curr = T_curr(1:3,1:3);
            t_curr = T_curr(1:3,4)';
            
            R_new = R_align * R_curr;
            t_new = s_align * (t_curr * R_align) + t_align;
            
            T_new = eye(4);
            T_new(1:3,1:3) = R_new;
            T_new(1:3,4) = t_new';
            vo_pose_aligned(k) = rigidtform3d(T_new);
        end
        
        abs_rotation_errors_deg = nan(minL, 1);
        
        for k = 1:minL
            R_gt = gt_pose_eval(1:3,1:3,k);
            R_est = vo_pose_aligned(k).A(1:3,1:3);
            R_diff = R_gt * R_est';
            R_diff = projectToSO3(R_diff);
            rot_ang_rad = rotationAngleFromMatrix(R_diff);
            abs_rotation_errors_deg(k) = rad2deg(abs(rot_ang_rad));
        end
        
        abs_rot_rmse_deg = sqrt(mean(abs_rotation_errors_deg.^2, 'omitnan'));
        abs_rot_mean_deg = mean(abs_rotation_errors_deg, 'omitnan');
        abs_rot_median_deg = median(abs_rotation_errors_deg, 'omitnan');
        abs_rot_std_deg = std(abs_rotation_errors_deg, 'omitnan');
        abs_rot_max_deg = max(abs_rotation_errors_deg, [], 'omitnan');
        
        rpe_translation_errors = [];
        rpe_rotation_errors_deg = [];
        
        for k = 1:(minL - RPE_DELTA)
            T_gt_1 = gt_pose_eval(:,:,k);
            T_gt_2 = gt_pose_eval(:,:,k + RPE_DELTA);
            T_est_1 = vo_pose_aligned(k).A;
            T_est_2 = vo_pose_aligned(k + RPE_DELTA).A;
            
            T_gt_rel = inv(T_gt_1) * T_gt_2;
            T_est_rel = inv(T_est_1) * T_est_2;
            T_err_rel = inv(T_gt_rel) * T_est_rel;
            
            trans_err_rel = norm(T_err_rel(1:3,4));
            R_err_rel = T_err_rel(1:3,1:3);
            R_err_rel = projectToSO3(R_err_rel);
            rot_err_rel_rad = rotationAngleFromMatrix(R_err_rel);
            rot_err_rel_deg = rad2deg(abs(rot_err_rel_rad));
            
            rpe_translation_errors(end+1,1) = trans_err_rel; %#ok<AGROW>
            rpe_rotation_errors_deg(end+1,1) = rot_err_rel_deg; %#ok<AGROW>
        end
        
        if isempty(rpe_translation_errors)
            rpe_trans_rmse = NaN;
            rpe_trans_mean = NaN;
            rpe_trans_median = NaN;
            rpe_trans_std = NaN;
            rpe_trans_max = NaN;
            
            rpe_rot_rmse_deg = NaN;
            rpe_rot_mean_deg = NaN;
            rpe_rot_median_deg = NaN;
            rpe_rot_std_deg = NaN;
            rpe_rot_max_deg = NaN;
        else
            rpe_trans_rmse = sqrt(mean(rpe_translation_errors.^2, 'omitnan'));
            rpe_trans_mean = mean(rpe_translation_errors, 'omitnan');
            rpe_trans_median = median(rpe_translation_errors, 'omitnan');
            rpe_trans_std = std(rpe_translation_errors, 'omitnan');
            rpe_trans_max = max(rpe_translation_errors, [], 'omitnan');
            
            rpe_rot_rmse_deg = sqrt(mean(rpe_rotation_errors_deg.^2, 'omitnan'));
            rpe_rot_mean_deg = mean(rpe_rotation_errors_deg, 'omitnan');
            rpe_rot_median_deg = median(rpe_rotation_errors_deg, 'omitnan');
            rpe_rot_std_deg = std(rpe_rotation_errors_deg, 'omitnan');
            rpe_rot_max_deg = max(rpe_rotation_errors_deg, [], 'omitnan');
        end
        
        kitti_trans_percent_errors = [];
        kitti_rot_deg_per_m_errors = [];
        
        gt_xyz = zeros(minL, 3);
        est_xyz = zeros(minL, 3);
        for k = 1:minL
            gt_xyz(k,:) = gt_pose_eval(1:3,4,k)';
            est_xyz(k,:) = vo_pose_aligned(k).Translation;
        end
        
        gt_cumdist = [0; cumsum(vecnorm(diff(gt_xyz,1,1),2,2))];
        
        for seg_idx = 1:numel(KITTI_SEGMENT_LENGTHS)
            seg_len_frames = KITTI_SEGMENT_LENGTHS(seg_idx);
            for k = 1:(minL - seg_len_frames)
                j = k + seg_len_frames;
                
                T_gt_1 = gt_pose_eval(:,:,k);
                T_gt_2 = gt_pose_eval(:,:,j);
                T_est_1 = vo_pose_aligned(k).A;
                T_est_2 = vo_pose_aligned(j).A;
                
                T_gt_rel = inv(T_gt_1) * T_gt_2;
                T_est_rel = inv(T_est_1) * T_est_2;
                T_err_rel = inv(T_gt_rel) * T_est_rel;
                
                gt_segment_length_m = gt_cumdist(j) - gt_cumdist(k);
                
                if gt_segment_length_m <= eps
                    continue;
                end
                
                trans_err_seg = norm(T_err_rel(1:3,4));
                R_err_seg = T_err_rel(1:3,1:3);
                R_err_seg = projectToSO3(R_err_seg);
                rot_err_seg_rad = rotationAngleFromMatrix(R_err_seg);
                rot_err_seg_deg = rad2deg(abs(rot_err_seg_rad));
                
                trans_percent = (trans_err_seg / gt_segment_length_m) * 100;
                rot_deg_per_m = rot_err_seg_deg / gt_segment_length_m;
                
                kitti_trans_percent_errors(end+1,1) = trans_percent; %#ok<AGROW>
                kitti_rot_deg_per_m_errors(end+1,1) = rot_deg_per_m; %#ok<AGROW>
            end
        end
        
        if isempty(kitti_trans_percent_errors)
            kitti_trans_percent_mean = NaN;
            kitti_trans_percent_median = NaN;
            kitti_trans_percent_std = NaN;
            kitti_rot_deg_per_m_mean = NaN;
            kitti_rot_deg_per_m_median = NaN;
            kitti_rot_deg_per_m_std = NaN;
        else
            kitti_trans_percent_mean = mean(kitti_trans_percent_errors, 'omitnan');
            kitti_trans_percent_median = median(kitti_trans_percent_errors, 'omitnan');
            kitti_trans_percent_std = std(kitti_trans_percent_errors, 'omitnan');
            
            kitti_rot_deg_per_m_mean = mean(kitti_rot_deg_per_m_errors, 'omitnan');
            kitti_rot_deg_per_m_median = median(kitti_rot_deg_per_m_errors, 'omitnan');
            kitti_rot_deg_per_m_std = std(kitti_rot_deg_per_m_errors, 'omitnan');
        end
        
        valid_pose_ratio = mean(is_pose_valid(1:minL));
        moved_per_frame = [0; vecnorm(diff(vo_pos(1:minL,:),1,1),2,2)];
        moved_frame_ratio = mean(moved_per_frame > 0);
        mean_matches = mean(match_count_per_frame(1:minL), 'omitnan');
        mean_inliers = mean(inlier_count_per_frame(1:minL), 'omitnan');
        
        newRow = table({currSet.name}, rmse, mean_err, max_err, transform.b, ...
            median_err, std_err, ...
            abs_rot_rmse_deg, abs_rot_mean_deg, abs_rot_median_deg, abs_rot_std_deg, abs_rot_max_deg, ...
            rpe_trans_rmse, rpe_trans_mean, rpe_trans_median, rpe_trans_std, rpe_trans_max, ...
            rpe_rot_rmse_deg, rpe_rot_mean_deg, rpe_rot_median_deg, rpe_rot_std_deg, rpe_rot_max_deg, ...
            kitti_trans_percent_mean, kitti_trans_percent_median, kitti_trans_percent_std, ...
            kitti_rot_deg_per_m_mean, kitti_rot_deg_per_m_median, kitti_rot_deg_per_m_std, ...
            valid_pose_ratio, moved_frame_ratio, mean_matches, mean_inliers, ...
            'VariableNames', {'Set', 'RMSE', 'Mean_Error', 'Max_Error', 'Scale', ...
            'Median_Error', 'Std_Error', ...
            'Abs_Rot_RMSE_deg', 'Abs_Rot_Mean_deg', 'Abs_Rot_Median_deg', 'Abs_Rot_Std_deg', 'Abs_Rot_Max_deg', ...
            'RPE_Trans_RMSE', 'RPE_Trans_Mean', 'RPE_Trans_Median', 'RPE_Trans_Std', 'RPE_Trans_Max', ...
            'RPE_Rot_RMSE_deg', 'RPE_Rot_Mean_deg', 'RPE_Rot_Median_deg', 'RPE_Rot_Std_deg', 'RPE_Rot_Max_deg', ...
            'KITTI_Trans_Percent_Mean', 'KITTI_Trans_Percent_Median', 'KITTI_Trans_Percent_Std', ...
            'KITTI_Rot_deg_per_m_Mean', 'KITTI_Rot_deg_per_m_Median', 'KITTI_Rot_deg_per_m_Std', ...
            'Valid_Pose_Ratio', 'Moved_Frame_Ratio', 'Mean_Matches', 'Mean_Inliers'});
        resultsTable = [resultsTable; newRow];
        
        plot3(vo_aligned(:,1), vo_aligned(:,2), vo_aligned(:,3), ...
            'Color', setColors(s), 'LineWidth', 1.8, 'DisplayName', currSet.name);
            
        fprintf('   ATE RMSE: %.4f | Median: %.4f | Std: %.4f\n', rmse, median_err, std_err);
        fprintf('   Abs Rot RMSE: %.4f deg | Mean: %.4f deg\n', abs_rot_rmse_deg, abs_rot_mean_deg);
        fprintf('   RPE Trans RMSE (delta=%d): %.4f | RPE Rot RMSE: %.4f deg\n', RPE_DELTA, rpe_trans_rmse, rpe_rot_rmse_deg);
        fprintf('   KITTI Drift: Trans %.4f %% | Rot %.6f deg/m\n', kitti_trans_percent_mean, kitti_rot_deg_per_m_mean);
        fprintf('   Valid Pose Ratio: %.4f | Moved Frame Ratio: %.4f\n', valid_pose_ratio, moved_frame_ratio);
    end
end

legend('show', 'Location', 'northeastoutside');
hold off;

fprintf('\n--- FINAL TRAJECTORY METRICS SUMMARY ---\n');
disp(sets(2).filter);
disp(resultsTable);

%% Updated Helper Function
function [pts, feats] = extractLocalFeatures(I, type)
    feats = []; pts = [];
    switch upper(type)
        case 'SURF'
            pts = detectSURFFeatures(I, 'MetricThreshold', 200);
            if ~isempty(pts), [feats, pts] = extractFeatures(I, pts); end
        case 'KAZE'
            pts = detectKAZEFeatures(I, 'Threshold', 0.001, 'NumOctaves', 4);
            if ~isempty(pts), [feats, pts] = extractFeatures(I, pts); end
        case 'ORB'
            pts = detectORBFeatures(I);
            if ~isempty(pts), [feats, pts] = extractFeatures(I, pts); end
        case 'BRISK'
            pts = detectBRISKFeatures(I);
            if ~isempty(pts), [feats, pts] = extractFeatures(I, pts); end
        otherwise
            error('Unknown detector type.');
    end
end

%% Added Helper Function
function R_out = projectToSO3(R_in)
    [U, ~, V] = svd(R_in);
    R_out = U * V';
    if det(R_out) < 0
        U(:,3) = -U(:,3);
        R_out = U * V';
    end
end

%% Added Helper Function
function theta = rotationAngleFromMatrix(R_in)
    R_in = projectToSO3(R_in);
    cos_theta = (trace(R_in) - 1) / 2;
    cos_theta = max(min(cos_theta, 1), -1);
    theta = acos(cos_theta);
end

%% Added Helper Function
function mask = createValidImageMask(I, borderFraction)
    [h, w, ~] = size(I);
    cx = (w + 1) / 2;
    cy = (h + 1) / 2;
    radius = borderFraction * 0.5 * min(h, w);
    
    [X, Y] = meshgrid(1:w, 1:h);
    mask = ((X - cx).^2 + (Y - cy).^2) <= radius^2;
end

%% Added Helper Function
function Iout = applyImageMask(Iin, mask)
    Iout = Iin;
    if size(Iin, 3) == 1
        Iout(~mask) = 0;
    else
        for c = 1:size(Iin, 3)
            channel = Iout(:,:,c);
            channel(~mask) = 0;
            Iout(:,:,c) = channel;
        end
    end
end

%% Added Helper Function
function [ptsOut, featsOut] = filterPointsAndFeaturesByMask(ptsIn, featsIn, mask)
    ptsOut = ptsIn;
    featsOut = featsIn;
    
    if isempty(ptsIn) || isempty(featsIn)
        return;
    end
    
    loc = round(ptsIn.Location);
    x = loc(:,1);
    y = loc(:,2);
    
    valid = x >= 1 & x <= size(mask,2) & y >= 1 & y <= size(mask,1);
    linearIdx = sub2ind(size(mask), y(valid), x(valid));
    
    validMaskFlag = false(size(valid));
    validMaskFlag(valid) = mask(linearIdx);
    
    ptsOut = ptsIn(validMaskFlag);
    
    if isa(featsIn, 'binaryFeatures')
        featsOut = featsIn(validMaskFlag);
    else
        featsOut = featsIn(validMaskFlag, :);
    end
end