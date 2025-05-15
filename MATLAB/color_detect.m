% ---- Video File Capture ----
% Point to the saved video file
% videoPath = 'video.mp4';        % adjust if your file name differs
% vid = VideoReader(videoPath);         % loads the video

% Read and show the first frame
% frame = readFrame(vid);
% imshow(frame);
% title('First Video Frame');

% Part - 2
% ---- Convert to HSV & Mask Red ----
% Convert the RGB frame into HSV color space
% hsvFrame = rgb2hsv(frame);

% Split into Hue (color), Saturation, and Value channels
% H = hsvFrame(:,:,1);  
% S = hsvFrame(:,:,2);
% V = hsvFrame(:,:,3);

% Create a binary mask for red:
%   - Hue near 0 or 1 (red wraps around)
%   - High saturation (to avoid dull colors)
%   - Value above 0.2 (not too dark)
% rawMask = (H < 0.05 | H > 0.95) & (S > 0.5) & (V > 0.2);

% Show what the mask looks like
% figure;
% imshow(rawMask);
% title('Raw Red Mask — white = red parts');
% -----------------------------

% PART - 3
% ---- Clean Mask & Find Bottle ----
% Remove small blobs under 1,000 pixels
% cleanMask = bwareaopen(rawMask, 1000);

% Show the cleaned mask
% figure;
% imshow(cleanMask);
% title('Cleaned Mask — only big red shapes');

% Find connected regions in the clean mask
% props = regionprops(cleanMask, 'BoundingBox', 'Area');

% Choose the largest region (should be your bottle)
% areas = [props.Area];
% [~, idx] = max(areas);
% box = props(idx).BoundingBox;

% Draw the box on the original frame
% figure;
% imshow(frame);
% hold on;
% rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 3);
% title('Detected Bottle with Bounding Box');
% hold off;

% PART - 4
% ---- Clean Mask & Find Bottle ----
% Remove small blobs under 1,000 pixels
cleanMask = bwareaopen(rawMask, 1000);

% Show the cleaned mask
figure;
imshow(cleanMask);
title('Cleaned Mask — only big red shapes');

% Find connected regions in the clean mask
props = regionprops(cleanMask, 'BoundingBox', 'Area');

% Choose the largest region (should be your bottle)
areas = [props.Area];
[~, idx] = max(areas);
box = props(idx).BoundingBox;

% Draw the box on the original frame
figure;
imshow(frame);
hold on;
rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 3);
title('Detected Bottle with Bounding Box');

% ---- Display Count ----
% How many blobs did we detect? (should be 1)
count = numel(props);           

% Prepare the text label
label = sprintf('Count: %d', count);

% Position it slightly above the box
x = box(1);
y = box(2) - 15;                

% Draw the text on the image
text(x, y, label, ...
     'Color', 'yellow', ...
     'FontSize', 14, ...
     'FontWeight', 'bold');
% --------------------------

hold off;




% ---- Enhanced Mask Cleaning ----
% 1) Remove tiny noise first
mask1 = bwareaopen(rawMask, 1000);

% 2) Close small holes and gaps with a disk‐shaped structuring element
se = strel('disk', 10);            % you can tweak the radius (10)
mask2 = imclose(mask1, se);

% 3) Fill any remaining holes inside the bottle
mask3 = imfill(mask2, 'holes');

% 4) Final noise removal just in case
cleanMask = bwareaopen(mask3, 1000);

% Show each step if you like (comment out after it works)
figure; imshow(mask1);   title('After bwareaopen');
figure; imshow(mask2);   title('After imclose');
figure; imshow(mask3);   title('After imfill');
figure; imshow(cleanMask); title('Final Clean Mask');




% ---- Display Count on Original Frame ----
% count = numel(props);  % Number of detected blobs (should be 1)

% ---- Display Final Output with Clean Count ----
if ~isempty(props)
    areas = [props.Area];
    [~, idx] = max(areas);
    box = props(idx).BoundingBox;

    % Draw the result
    figure;
    imshow(frame);
    hold on;
    rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 3);

    % Force count to 1 since we only use the largest blob
    label = 'Count: 1';
    x = box(1); y = box(2) - 15;
    text(x, y, label, 'Color', 'yellow', 'FontSize', 14, 'FontWeight', 'bold');
    title('Final Output with Clean Count');
    hold off;
else
    disp('No red object found!');
end





% ---- Video Processing with Saving and Progress Bar ----

% Specify the input and output video files
inputVideoFile = 'video.mp4';  % Replace with your actual video file name
outputVideoFile = 'video_output.mp4';

% Create VideoReader and VideoWriter objects
v = VideoReader(inputVideoFile);
outputVideo = VideoWriter(outputVideoFile, 'MPEG-4');
outputVideo.FrameRate = v.FrameRate;
open(outputVideo);

% Determine the total number of frames
totalFrames = floor(v.Duration * v.FrameRate);

% Create a progress bar
h = waitbar(0, 'Processing video...');

% Process each frame
for frameIdx = 1:totalFrames
    frame = readFrame(v);

    % Convert RGB frame to HSV
    hsvFrame = rgb2hsv(frame);
    H = hsvFrame(:,:,1);
    S = hsvFrame(:,:,2);
    V = hsvFrame(:,:,3);

    % Create a binary mask for red color
    rawMask = (H < 0.05 | H > 0.95) & (S > 0.5) & (V > 0.2);

    % Clean the mask
    mask1 = bwareaopen(rawMask, 1000);
    se = strel('disk', 10);
    mask2 = imclose(mask1, se);
    mask3 = imfill(mask2, 'holes');
    cleanMask = bwareaopen(mask3, 1000);

    % Find connected components
    props = regionprops(cleanMask, 'BoundingBox', 'Area');

    if ~isempty(props)
        % Find the largest blob
        areas = [props.Area];
        [~, idx] = max(areas);
        box = props(idx).BoundingBox;

        % Draw bounding box and label on the frame
        frame = insertShape(frame, 'Rectangle', box, 'Color', 'green', 'LineWidth', 3);
        position = [box(1), box(2) - 15];
        frame = insertText(frame, position, 'Count: 1', 'FontSize', 14, 'BoxColor', 'yellow', 'TextColor', 'black');
    end

    % Write the processed frame to the output video
    writeVideo(outputVideo, frame);

    % Update the progress bar
    waitbar(frameIdx / totalFrames, h, sprintf('Processing frame %d of %d...', frameIdx, totalFrames));
end

% Close the VideoWriter and progress bar
close(outputVideo);
close(h);

% Notify that processing is complete
disp('✅ Video processing complete! Processed video saved as:');
disp(outputVideoFile);

