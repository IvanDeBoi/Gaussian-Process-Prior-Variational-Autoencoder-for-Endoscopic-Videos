clc
% Configuration
videoFile = 'Hangar.mp4'; 
outputFolder = 'extracted_frames_lowres';
targetRes = 256; % Final output size (256x256)

% --- Square Crop Parameters ---
% Define the top-left corner of the square you want to extract
cropX = 400; % Horizontal start
cropY = 50;  % Vertical start
squareSize = 1000; % How many pixels wide/high the crop should be (e.g., 400x400)

% --- Range Parameters ---
startFrame = 88;
endFrame   = 3375%3477;

if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
v = VideoReader(videoFile);

% Access properties directly
nativeWidth  = v.Width;
nativeHeight = v.Height;

% Display in command window
fprintf('Native Resolution: %d x %d\n', nativeWidth, nativeHeight);

fprintf('Total frames in video: %d\n', v.NumFrames);
%resStr = sprintf('%dx%d', targetRes, targetRes);
resStr = sprintf('%d', targetRes);

% Loop through the specified range
for frameIdx = startFrame : endFrame
    img = read(v, frameIdx);
    [h, w, ~] = size(img);
    
    % --- Apply Square Cropping ---
    % Define the square coordinates
    yRange = cropY : (cropY + squareSize - 1);
    xRange = cropX : (cropX + squareSize - 1);
    
    % Safety check: Ensure the square doesn't go off the edge of the video
    if max(yRange) > h || max(xRange) > w
        error('Square crop (size %d at %d,%d) exceeds video dimensions (%d_%d).', ...
               squareSize, cropX, cropY, w, h);
    end
    
    % Extract the square (Pixels are cut off outside this range)
    imgSquare = img(yRange, xRange, :);
    
    % --- Resize the Square ---
    % Since the source is already a square, this won't squash it
    imgResized = imresize(imgSquare, [targetRes, targetRes]);
    
    % Save
    %EndoSLAM_From_Unity_320_002
    filename = sprintf('Hangar_%s_%06d.png', resStr, frameIdx);
    imwrite(imgResized, fullfile(outputFolder, filename));
    
    fprintf('Processed frame %d: Extracted %dx%d square -> Resized to %s\n', ...
            frameIdx, squareSize, squareSize, resStr);
end