%% color_detect.m
% Real-Time Color-Based Object Detection & Counting
% Uses predefined HSV ranges to detect multiple colors in a video,
% draws color-coded bounding boxes, overlays counts, logs data, and
% writes a processed video.

%% Step 1 – Define Color HSV Ranges & RGB Map
% (run this first to set up your color definitions)
colors(1).name = 'Red';    colors(1).hsv  = [0.95 1.0; 0 1; 0.2 1];
colors(2).name = 'Green';  colors(2).hsv  = [0.25 0.40; 0.5 1; 0.2 1];
colors(3).name = 'Yellow'; colors(3).hsv  = [0.10 0.18; 0.5 1; 0.2 1];
colors(4).name = 'Blue';   colors(4).hsv  = [0.55 0.75; 0.5 1; 0.2 1];
colors(5).name = 'Pink';   colors(5).hsv  = [0.85 1.0; 0.5 1; 0.2 1];
colors(6).name = 'Orange'; colors(6).hsv  = [0.05 0.10; 0.5 1; 0.2 1];
colors(7).name = 'Purple'; colors(7).hsv  = [0.70 0.85; 0.5 1; 0.2 1];
nColors = numel(colors);

% RGB triplets (0–1 range) for drawing boxes
rgbMap = struct( ...
  'Red',    [1 0   0], ...
  'Green',  [0 1   0], ...
  'Yellow', [1 1   0], ...
  'Blue',   [0 0   1], ...
  'Pink',   [1 0.4 0.6], ...
  'Orange', [1 0.5 0], ...
  'Purple', [0.5 0  0.5] ...
);

%% Step 2 – Let User Select Which Colors to Detect
% (run this to choose your SKUs at runtime)
fprintf('Available colors:\n');
for i = 1:nColors
    fprintf('  %d) %s\n', i, colors(i).name);
end
choices = input('Enter indices of colors to detect (e.g. [1 3 5]): ');

%% Step 3 – Prepare Video I/O & Logging
% (run this next to open files and start the progress bar)
inputVideoFile  = 'fruit-and-vegetable-detection.mp4';          % your source video
outputVideoFile = 'video_output_fruit.mp4';   % annotated output
logFile         = 'detection_log_fruit.csv';  % CSV log

v = VideoReader(inputVideoFile);
vw = VideoWriter(outputVideoFile, 'MPEG-4');
vw.FrameRate = v.FrameRate;
open(vw);

totalFrames = floor(v.Duration * v.FrameRate);
hWait = waitbar(0, 'Processing video...');

% create log with headers
fid = fopen(logFile, 'w');
fprintf(fid, 'Frame');
for k = 1:numel(choices)
    fprintf(fid, ',%s', colors(choices(k)).name);
end
fprintf(fid, '\n');

%% Step 4 – Main Processing Loop
% (run this block to process & annotate the entire video)
for frameIdx = 1:totalFrames
    frame = readFrame(v);
    
    % convert to HSV
    hsvFrame = rgb2hsv(frame);
    H = hsvFrame(:,:,1);
    S = hsvFrame(:,:,2);
    V = hsvFrame(:,:,3);
    
    % reset per-frame counts
    frameCounts = zeros(1, numel(choices));

    % detect each selected color
    for c = 1:numel(choices)
        idx = choices(c);
        thr = colors(idx).hsv;
        
        % hue mask (special wrap for red)
        if strcmp(colors(idx).name, 'Red')
            hueMask = (H < 0.05) | (H > 0.95);
        else
            hueMask = (H >= thr(1,1)) & (H <= thr(1,2));
        end
        satMask = (S >= thr(2,1)) & (S <= thr(2,2));
        valMask = (V >= thr(3,1)) & (V <= thr(3,2));
        rawMask = hueMask & satMask & valMask;

        % clean up the mask
        mask1 = bwareaopen(rawMask, 1000);
        mask2 = imclose(mask1, strel('disk',10));
        mask3 = imfill(mask2, 'holes');
        cleanMask = bwareaopen(mask3, 1000);

        % find blobs
        props = regionprops(cleanMask, 'BoundingBox');
        frameCounts(c) = numel(props);

        % draw each blob
        for j = 1:frameCounts(c)
            box = props(j).BoundingBox;
            col = rgbMap.(colors(idx).name);
            frame = insertShape(frame, ...
                'Rectangle', box, ...
                'Color',    col, ...
                'LineWidth',3);
        end
    end

    % overlay total counts in top-left corner
    y0 = 10;
    for c = 1:numel(choices)
        txt = sprintf('%s: %d', colors(choices(c)).name, frameCounts(c));
        frame = insertText(frame, [10,y0], txt, ...
            'FontSize',14, 'BoxColor','black', 'TextColor','white');
        y0 = y0 + 20;
    end

    % write and update
    writeVideo(vw, frame);
    waitbar(frameIdx/totalFrames, hWait, ...
        sprintf('Processing frame %d of %d...', frameIdx, totalFrames));
    
    % log counts
    fprintf(fid, '%d', frameIdx);
    for c = 1:numel(choices)
        fprintf(fid, ',%d', frameCounts(c));
    end
    fprintf(fid, '\n');
end

%% Step 5 – Cleanup & Notify
% (run last to close files and finish up)
close(vw);
close(hWait);
fclose(fid);
disp('✅ Done! Output video and log file are ready.');
