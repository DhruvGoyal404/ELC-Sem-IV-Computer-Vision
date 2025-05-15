% Step 1: Setup
inputVideoFile = '16453-272487468_small.mp4';  % Replace with your actual video file name
outputVideoFile = 'video_output_3.mp4';
logFile = 'detection_log_3.csv';

% Create VideoReader and VideoWriter objects
v = VideoReader(inputVideoFile);
outputVideo = VideoWriter(outputVideoFile, 'MPEG-4');
outputVideo.FrameRate = v.FrameRate;
open(outputVideo);

% Determine the total number of frames
totalFrames = floor(v.Duration * v.FrameRate);

% Create a progress bar
h = waitbar(0, 'Processing video...');

% Open the log file for writing
fid = fopen(logFile, 'w');
fprintf(fid, 'Frame,Count\n');

% Step 2: Process each frame
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

    % Count the number of detected objects
    count = numel(props);

    % Draw bounding boxes and labels on the frame
    for i = 1:count
        box = props(i).BoundingBox;
        frame = insertShape(frame, 'Rectangle', box, 'Color', 'green', 'LineWidth', 3);
        position = [box(1), box(2) - 15];
        label = sprintf('Object %d', i);
        frame = insertText(frame, position, label, 'FontSize', 14, 'BoxColor', 'yellow', 'TextColor', 'black');
    end

    % Display the total count on the frame
    countLabel = sprintf('Count: %d', count);
    frame = insertText(frame, [10, 10], countLabel, 'FontSize', 14, 'BoxColor', 'red', 'TextColor', 'white');

    % Write the processed frame to the output video
    writeVideo(outputVideo, frame);

    % Log the count to the CSV file
    fprintf(fid, '%d,%d\n', frameIdx, count);

    % Update the progress bar
    waitbar(frameIdx / totalFrames, h, sprintf('Processing frame %d of %d...', frameIdx, totalFrames));
end

% Close the VideoWriter, progress bar, and log file
close(outputVideo);
close(h);
fclose(fid);

% Notify that processing is complete
disp('✅ Video processing complete! Processed video saved as:');
disp(outputVideoFile);
disp('Detection log saved as:');
disp(logFile);
