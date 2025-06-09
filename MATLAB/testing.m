%% color_detect_final.m
% Layered Preprocessing + Robust Color-Based Detection & Counting

%% Step 1 – Define Color Ranges & Display Colors
fprintf('Step 1: Defining HSV ranges and display colors...\n');
colDefs = {
  'Red',        [0.95 0.05; 0.70 1.00; 0.50 1.00], [1 0   0];
  'Green',      [0.20 0.40; 0.40 1.00; 0.20 1.00], [0 1   0];
  'Yellow',     [0.10 0.18; 0.50 1.00; 0.50 1.00], [1 1   0];
  'Blue',       [0.55 0.75; 0.30 1.00; 0.20 1.00], [0 0   1];
  'Pink',       [0.88 0.94; 0.30 1.00; 0.40 1.00], [1 0.4 0.6];
  'Orange',     [0.05 0.10; 0.50 1.00; 0.40 1.00], [1 0.5 0];
  'Purple',     [0.70 0.85; 0.30 1.00; 0.30 1.00], [0.5 0   0.5];
  'DarkGreen',  [0.22 0.40; 0.40 1.00; 0.10 0.50], [0 0.5 0];
  'LightBlue',  [0.48 0.64; 0.20 0.80; 0.50 1.00], [0.5 0.8 1];
  'Brown',      [0.05 0.15; 0.20 0.70; 0.10 0.60], [0.6 0.3 0.1];

  %–– New Entries –––
  'Cyan',       [0.45 0.55; 0.40 1.00; 0.50 1.00], [0 1 1];       % pure cyan at 180° hue :contentReference[oaicite:4]{index=4}
  'Magenta',    [0.80 0.95; 0.40 1.00; 0.50 1.00], [1 0 1];       % pure magenta at 300° hue :contentReference[oaicite:5]{index=5}
  'Black',      [0.00 1.00; 0.00 1.00; 0.00 0.20], [0 0 0];       % very low V values :contentReference[oaicite:6]{index=6}
  'White',      [0.00 1.00; 0.00 0.20; 0.80 1.00], [1 1 1];       % very low S, high V :contentReference[oaicite:7]{index=7}
  'Gray',       [0.00 1.00; 0.00 0.20; 0.20 0.80], [0.5 0.5 0.5]; % low S, mid V :contentReference[oaicite:8]{index=8}
  'Teal',       [0.45 0.55; 0.40 1.00; 0.20 0.60], [0 0.5 0.5];   % dark cyan variant :contentReference[oaicite:9]{index=9}
  'Navy',       [0.58 0.65; 0.50 1.00; 0.20 0.60], [0 0 0.5];     % dark blue variant :contentReference[oaicite:10]{index=10}
  'Olive',      [0.10 0.18; 0.40 1.00; 0.20 0.60], [0.5 0.5 0];   % dark yellow–green :contentReference[oaicite:11]{index=11}
  'Maroon',     [0.95 1.00; 0.40 1.00; 0.20 0.60], [0.5 0 0];     % dark red variant :contentReference[oaicite:12]{index=12}
  'Aqua',       [0.45 0.55; 0.40 1.00; 0.50 1.00], [0 1 1];       % synonym for cyan :contentReference[oaicite:13]{index=13}
};

nColors = size(colDefs,1);
for i=1:nColors
  colors(i).name = colDefs{i,1};
  colors(i).hsv  = colDefs{i,2};
  colors(i).rgb  = colDefs{i,3};
end
fprintf('  %d colors configured.\n', nColors);

%% Step 2 – Video Setup & Frame Count
fprintf('Step 2: Preparing video I/O...\n');
videoPath       = 'fruit-and-vegetable-detection.mp4';
outputVideoFile = 'june_final_output_pakka.mp4';
logFile         = 'june_final_log_pakka.csv';

v = VideoReader(videoPath);
fprintf('  Input video: %s (%.1fs @ %.2f fps)\n', videoPath, v.Duration, v.FrameRate);

% Estimate total frames
totalFrames = floor(v.Duration * v.FrameRate);
fprintf('  Estimated frames: %d\n', totalFrames);

%% Step 3 – Init Writer, Logger, Waitbar
v.CurrentTime = 0;  
vw = VideoWriter(outputVideoFile,'MPEG-4'); vw.FrameRate=v.FrameRate; open(vw);
fprintf('  Output video: %s\n', outputVideoFile);

fid = fopen(logFile,'w');
fprintf(fid,'Frame');
for i=1:nColors, fprintf(fid,',%s',colors(i).name); end
fprintf(fid,'\n');
fprintf('  Logging to: %s\n', logFile);

hWait = waitbar(0,'Processing...'); fprintf('Step 3: Started.\n');

%% Step 4 – Main Loop with Advanced Cleaning
frameIdx = 0;
while hasFrame(v)
  frameIdx = frameIdx + 1;
  frameOrig = readFrame(v);

  % 1) Pre-smooth to reduce noise/highlight
  frame = medfilt3(frameOrig, [5 5 1]);  % median filter spatially

  % 2) Convert to HSV
  hsvF = rgb2hsv(frame);
  H = hsvF(:,:,1); S = hsvF(:,:,2); V = hsvF(:,:,3);

  % Prepare counts and annotation
  counts = zeros(1,nColors);
  annotated = frameOrig;

  for c = 1:nColors
    thr = colors(c).hsv;
    % Hue mask
    if strcmp(colors(c).name,'Red')
      hueM = (H<thr(1,2)) | (H>thr(1,1));
    else
      hueM = (H>=thr(1,1)) & (H<=thr(1,2));
    end
    satM = (S>=thr(2,1)) & (S<=thr(2,2));
    valM = (V>=thr(3,1)) & (V<=thr(3,2));
    rawMask = hueM & satM & valM;

    % Morphological pipeline
    clean = imopen(rawMask, strel('disk',5));          % remove tiny, smooth edges
    clean = imclose(clean, strel('disk',10));         % close gaps
    clean = imfill(clean,'holes');                    % fill holes
    clean = bwareaopen(clean, 1000);                  % final speck removal

    % Extract and draw
    props = regionprops(clean,'BoundingBox');
    counts(c) = numel(props);
    for j=1:counts(c)
      bb = props(j).BoundingBox;
      annotated = insertShape(annotated,'Rectangle',bb,'Color',colors(c).rgb,'LineWidth',3);
      annotated = insertText(annotated,bb(1:2),colors(c).name,'FontSize',12,...
        'BoxColor',colors(c).rgb,'TextColor','white','BoxOpacity',0.7);
    end
  end

  % 3) Draw dynamic total-count panel
  panel = repmat(uint8([30 30 30]),[nColors+1,1]); % gray boxes
  txtLines = ['Total: ' num2str(sum(counts))];
  for c=1:nColors
    panel = [panel; uint8(255*colors(c).rgb)];
  end
  % overlay text
  y0 = 5;
  annotated = insertText(annotated,[5 y0], txtLines,'FontSize',14,'BoxColor','black','TextColor','white');
  for c=1:nColors
    y0=y0+20;
    annotated = insertText(annotated,[5 y0], sprintf('%s: %d',colors(c).name,counts(c)),...
      'FontSize',12,'BoxColor','black','TextColor','white');
  end

  % 4) Write & log
  writeVideo(vw, annotated);
  fprintf(fid,'%d',frameIdx);
  fprintf(fid,',%d',counts);
  fprintf(fid,'\n');

  % 5) Progress
  waitbar(frameIdx/totalFrames,hWait,sprintf('Frame %d/%d',frameIdx,totalFrames));
  drawnow;
end

%% Step 5 – Cleanup
close(vw); fclose(fid); if ishandle(hWait), close(hWait); end
fprintf('Step 5: Done! Output → %s, Log → %s\n', outputVideoFile, logFile);
