# Assignment 2  
## Eulerian Video Magnification to reveal temporal image variations  
### Initials and color transformation

먼저 laplacian의 level과 적용할 gaussian의 standard deviation을 설정하였다.  
그런 뒤 matlab의 VideoReader 함수를 통해 data 폴더 안에 있는 영상을 불러오고,  
영상 파일의 height, width, frames 그리고 frame rate를 확인하였다.  
영상을 간단하게 확인한 후, readFrame을 통해 영상의 각 frame에 접근하였다.  
각 frame을 im2double을 통해 pixel 값을 range가 0~1인 double형으로 변환하고,  
rgb2ntsc를 통해 rgb의 이미지를 YIQ space로 변환하였다.
```matlab
%% Initials and color transformation
level = 4; %face, own
%level = 3; %baby2
std = 1;

% Read video
video_path = './data/face.mp4'; %face
%video_path = './data/baby2.mp4'; %baby2
%video_path = './data/own.mp4'; %own

v = VideoReader(video_path);

height = v.Height;
width = v.Width;

frames = v.NumFrames;
video_fps = v.FrameRate;

convert_color = zeros(frames, height, width, 3);
for i=1:frames
    frame = readFrame(v);
    frame = im2double(frame);

    frame = rgb2ntsc(frame);
    
    convert_color(i, :, :, :) = frame;
end
```

### Laplacian pyramid  

Laplacian pyramid를 (frames, height, width, channels)를 가지는 4D array로 구성하였다.  
이때, width는 각 pyramid를 이어 붙였기 때문에 pyramid들의 width들의 합이다.  
Height는 원본의 height를 갖지만, 각 pyramid를 이어 붙였기에 옆으로 갈수록 height가 비어있게 된다.  
이러한 array는 laplacian pyramid라는 함수를 직접 구현하여 만들었다.  
```matlab
%% Laplacian pyramid
pyramid = laplacian_pyramid(squeeze(convert_color(1, :, :, :)), level, std);
[height, width, channels] = size(pyramid);
pyramid_frames = zeros(frames, height, width, channels);

for i=1:frames
    pyramid_frames(i, :, :, :) = laplacian_pyramid(squeeze(convert_color(i, :, :, :)), level, std);
end
```

아래가 구현한 laplacian pyramid 함수이다.  
imgaussfilt를 통해 미리 정한 standard deviation을 갖도록 gaussian blur를 하였다.  
그런 뒤, 이 이미지에 downsample을 하고 이를 다시 upsample하여 원래의 이미지와의 차를 구하였다.  
이렇게 laplacian pyramid를 만들고 그 이미지들을 이어 붙여 array를 만들었다.  
마지막에 남는 gaussian image도 맨 뒤에 추가하였다.

```matlab
function result = laplacian_pyramid(img, level, sigma)
    [height, width, ~] = size(img);
    
    width = ceil(width * (2 - 1/(2^(level-1))));
    
    result = ones(height, width, 3) * 0.5;

    origin_image = img;
    width_idx = 1;    
    for i=1:level-1
        gauss_image = imgaussfilt(origin_image, sigma);
        downsample_image = imresize(gauss_image, 0.5, 'Antialiasing', false);
        
        upsample_image = imresize(downsample_image, [size(gauss_image, 1), size(gauss_image, 2)], 'Antialiasing', false);
              
        res_image = origin_image - upsample_image;
        
        [res_height, res_width, ~] = size(res_image);

        result(1:res_height, width_idx:width_idx + res_width - 1, :) = res_image;
        
        width_idx = width_idx + res_width;
        origin_image = downsample_image;
    end

    [origin_height, origin_width, ~] = size(origin_image);
    result(1:origin_height, width_idx:width_idx + origin_width - 1, :) = origin_image;
end
```

### Temporal filtering & Extracting the frequency band of interest  

이 부분에서는 논문에 주어진 대로 alpha와 low pass, high pass frequency를 적용하여 filter를 구현하였다.  
속도와 fft로 얻어지는 결과로 인하여 frame의 time domain을 height, width, channel 마다 퓨리에 변환을 한 뒤,  
그 변환 값을 filtering하는 것으로 그 결과를 얻어냈다.  

```matlab
%% Temporal filtering and Extracting the frequency band of interest
Fs = video_fps;

padnum = 2 * frames;
dummy_size = size(pyramid_frames);
fft_pyramid = fft(pyramid_frames, padnum, 1);

addpath('./src');

Hd = butterworthBandpassFilter(Fs, 256, 0.8, 1.0); %face, own
%Hd = butterworthBandpassFilter(Fs, 256, 2.33, 2.67); %baby2
fftHd = freqz(Hd, frames + 1);

alpha = [0 0 0 100]; % for level 4
%alpha = [0 0 100]; % for level 3

alpha = alpha / sum(alpha) * 15;
[~, total_height, total_width, channels] = size(fft_pyramid);

alpha_mat = ones(total_height, total_width, channels);
alpha_mat(:, :, 1) = zeros(total_height, total_width);

width_idx = 1;
width_size = v.Width;
for i=1:level
    new_idx = width_idx+width_size;
    alpha_mat(:, width_idx:new_idx-1, :) = alpha(i) * alpha_mat(:, width_idx:new_idx-1, :);
    width_idx = new_idx;
    width_size = ceil(width_size/2);
end

fftHd_expand = zeros(padnum, 1);
fftHd_expand(1:frames+1) = fftHd;
fftHd_expand(frames+2:end) = fftHd(end-1:-1:2);

[~, height, width, channels] = size(fft_pyramid);
for channel=1:channels
    for w=1:width
        for h=1:height
            fft_pyramid(:, h, w, channel) = fft_pyramid(:, h, w, channel) .* (1 + fftHd_expand * alpha_mat(h, w, channel));
        end
    end
end

pyramid_frames = abs(ifft(fft_pyramid, padnum, 1));
```

### Image reconstruction  

Filtering을 통해 얻은 이미지를 아래와 같이 다시 영상으로 얻었다.  
collapse laplacian pyramid라는 직접 구현한 함수를 통해 filtering한 결과를 다시 합쳤다.  
그렇게 얻은 array를 영상으로 저장하기 전에 먼저 ntsc2rgb라는 함수를 통해 YIQ space에서 RGB space로 변환하고,  
그렇게 얻은 이미지가 원본 영상의 크기와 같아지도록 imresize를 해주었다.  
그런 뒤 VideoWriter와 writeVideo를 통해 영상을 저장하였다.  

```matlab
%% Image reconstruction
result_frames = zeros(frames, v.Height, v.Width, 3);
for i=1:frames
    temp = collapse_laplacian_pyramid(squeeze(pyramid_frames(i, :, :, :)), level, v.Height, v.Width);
    
    temp = ntsc2rgb(temp);
    
    temp = imresize(temp, [v.Height v.width], 'Antialiasing', true);
    
    result_frames(i, :, :, :) = temp;
end

out = VideoWriter('Result.avi');
open(out);

for i=1:frames
    writeVideo(out, squeeze(result_frames(i, :, :, :)));
end

close(out);
```

아래의 코드가 collapse laplacian pyramid이다.  
직접 구현한 get image 코드를 통해 주어진 level에 해당하는 이미지를 불어온 뒤,  
그 이미지를 upsample하여 이전 level의 이미지와 더해 원본의 이미지의 크기로 reconstruction하였다.  

```matlab
function result = collapse_laplacian_pyramid(pyramid, level, height, width)
    origin_image = get_image(pyramid, level, height, width);
    for i=1:level-1
        res_image = get_image(pyramid, level - i, height, width);
        upsample_image = imresize(origin_image, [size(res_image, 1), size(res_image, 2)], 'Antialiasing', false);
        origin_image = upsample_image + res_image;
    end
    
    result = origin_image;
end
```

아래는 level에 해당하는 이미지를 array로부터 불러오는 get image 함수이다.  

```matlab
function result = get_image(pyramid, index, height, width)
    
    length = 0;
    for i=1:index - 1
        length = length + width;
        width = ceil(width / 2);
        height = ceil(height / 2);
    end
    
    result = pyramid(1:height, length + 1:length + width, :);    
end
```

<table>
    <tr>
        <th>Grey world assumption</th>
        <th>White world assumption</th>
    </tr>
    <tr>
        <td><img src='./image/rgb_grey.png'></td>
        <td><img src='./image/rgb_white.png'></td>
    </tr>
</table>

### Capture and motion-magnify your own video  

직접 찍은 영상을 통해 eulerian video magnification을 진행하였다.  
여기서는 level과 filter를 face 영상 파일과 같게 설정한 뒤 그 결과를 확인하였다.  
