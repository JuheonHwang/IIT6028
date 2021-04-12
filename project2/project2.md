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

이 부분에서는 논문에 주어진 대로 $$ \alpha $$와 low pass, high pass frequency를 적용하여 filter를 구현하였다.  


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

alpha = [0 0 0 0 100];

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

그런 뒤, rggb인지 bggr인지 확인하기 위해 그 이미지를 출력하여 확인하였는데,  
rggb에서의 전체적인 이미지가 나타내는 색이 적절하여 rggb의 bayer pattern을 가진다는 것을 확인하였다.  
<table>
    <tr>
        <th>rggb</th>
        <th>bggr</th>
    </tr>
    <tr>
        <td><img src='./image/lin_first.png'></td>
        <td><img src='./image/lin_second.png'></td>
    </tr>
</table>

### White balancing  

이미지가 전체적으로 초록 빛을 띄고 있기 때문에 white balancing을 통해서 이미지를 자연스럽게 해주었다.  
White balancing은 green pixel의 값을 기준으로 red와 blue를 조정한다.  
#### Grey world assumption  

Grey world assumption은 이미지가 전체적으로 어둡다는 가정을 한다.  
rgb 세 채널의 각 mean 값을 가지고 이미지를 수정한다.
```matlab
lin_mean = mean(lin_rgb, [1 2]);
red_grey = lin1 .*(lin_mean(:, :, 2) / lin_mean(:, :, 1));
green_grey = green;
blue_grey = lin4 .*(lin_mean(:, :, 2) / lin_mean(:, :, 3));
rgb_grey = cat(3, red_grey, green_grey, blue_grey);

figure;
imshow(rgb_grey);
imwrite(rgb_grey, 'rgb_grey.png');
```

#### White world assumption  

White world assumption은 이미지가 전체적으로 밝다는 가정을 한다.  
rgb 세 채널의 각 max 값을 가지고 이미지를 수정한다.
```matlab
lin_max = max(lin_rgb, [], [1 2]);
red_white = lin1 .*(lin_max(:, :, 2) / lin_max(:, :, 1));
green_white = green;
blue_white = lin4 .*(lin_max(:, :, 2) / lin_max(:, :, 3));
rgb_white = cat(3, red_white, green_white, blue_white);

figure;
imshow(rgb_white);
imwrite(rgb_white, 'rgb_white.png');
```

아래의 표를 통해 grey world assumption과 white world assumption의 결과를 비교하였는데,  
그 결과가 grey world assumption이 더 보기에 적합한 것 같아서 이 뒤의 과정에서는 grey world의 결과를 사용하였다.  
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

### Demosaicing  

matlab의 interp2 함수를 통해 이미지 demosaic를 하였다.
```matlab
red_demosaic = interp2(red_grey);
green_demosaic = interp2(green_grey);
blue_demosaic = interp2(blue_grey);
rgb_demosaic = cat(3, red_demosaic, green_demosaic, blue_demosaic);

figure;
imshow(rgb_demosaic);
imwrite(rgb_demosaic, 'rgb_demosaic.png');
```

<table>
    <tr>
        <th>Demosaicing</th>
    </tr>
    <tr>
        <td><img src='./image/rgb_demosaic.png'></td>
    </tr>
</table>

### Brightness adjustment and gamma correction  
#### Brightness adjustment  

이미지의 밝기를 조절하기 위해 여러 값을 비교하였다.  
이미지를 3.5배 하여 밝기를 높여 좀 더 적절한 색을 가지는 이미지를 얻을 수 있었다.
```matlab
rgb_adj = rgb_demosaic * 3.5;
imshow(rgb_adj);
imwrite(rgb_adj, 'rgb_adj.png');
```

<table>
    <tr>
        <th>Brightness adjustment</th>
    </tr>
    <tr>
        <td><img src='./image/rgb_adj.png'></td>
    </tr>
</table>

#### Gamma correction  

Gamma correction을 통해 사람이 보기에 적절한 색을 가지도록 이미지를 조정하였다.  
Gamma correction의 기준 값은 grayscale의 0.0031308을 기준으로 다른 함수를 사용하도록 하였다.  
```matlab
grayscale_adj = rgb2gray(rgb_adj);
sz_adj = size(grayscale_adj);
non_linear = zeros(sz_adj(1), sz_adj(2), 3);

for n = 1:sz_adj(1)
    for m = 1:sz_adj(2)
        if grayscale_adj(n,m) < 0.0031308
            val = 12.92 * rgb_adj(n, m, :);
        else
            val = (1 + 0.055) * power(rgb_adj(n, m, :),1/2.4) - 0.055;
        end
        non_linear(n, m, :) = val;
    end
end

figure;
imshow(non_linear);
imwrite(non_linear, 'non_linear.png');
```

<table>
    <tr>
        <th>Gamma correction</th>
    </tr>
    <tr>
        <td><img src='./image/non_linear.png'></td>
    </tr>
</table>

### Compression  

여기서 압축률에 따른 결과를 비교하였다. 
```matlab
imwrite(non_linear, 'non_linear_95.jpeg', 'quality', 95);
imwrite(non_linear, 'non_linear_50.jpeg', 'quality', 50);
imwrite(non_linear, 'non_linear_30.jpeg', 'quality', 30);
imwrite(non_linear, 'non_linear_20.jpeg', 'quality', 20);
imwrite(non_linear, 'non_linear_15.jpeg', 'quality', 15);
imwrite(non_linear, 'non_linear_10.jpeg', 'quality', 10);
imwrite(non_linear, 'non_linear_5.jpeg', 'quality', 5);
```
오리지널 파일(PNG)과 JPEG 95로 압축한 파일을 봤을 때, 눈으로는 차이를 느끼지 못하였다.  
오리지널 파일(PNG)의 크기는 16,745,000 byte이고 JPEG 95의 크기는 3,324,321 byte이다.  
압축률은 3,324,321 / 16,745,000 = 0.1985이다.  

다른 파일들의 압축률에 대해서는,  
Quality 50: 797,045 / 16,745,000 = 0.0476  
Quality 30: 584,334 / 16,745,000 = 0.0349  
Quality 20: 464,095 / 16,745,000 = 0.0277  
Quality 15: 400,588 / 16,745,000 = 0.0239  
Quality 10: 329,990 / 16,745,000 = 0.0197  
Quality 5: 252,052 / 16,745,000 = 0.0151
임을 확인할 수 있었다.  

압축 Quality가 20까지는 원본 이미지와 큰 차이를 확인할 수 없었다.  
하지만 그보다 작은 quality에 대해서는 이미지가 왜곡된 것을 쉽게 확인할 수 있었다.  

<table>
    <tr>
        <th>PNG</th>
        <th>Quality 95</th>
    </tr>
    <tr>
        <td><img src='./image/non_linear.png'></td>
        <td><img src='./image/non_linear_95.jpeg'></td>
    </tr>
</table>

<table>
    <tr>
        <th>Quality 50</th>
        <th>Quality 30</th>
        <th>Quality 20</th>
    </tr>
    <tr>
        <td><img src='./image/non_linear_50.jpeg'></td>
        <td><img src='./image/non_linear_30.jpeg'></td>
        <td><img src='./image/non_linear_20.jpeg'></td>
    </tr>
</table>

<table>
    <tr>
        <th>Quality 15</th>
        <th>Quality 10</th>
        <th>Quality 5</th>
    </tr>
    <tr>
        <td><img src='./image/non_linear_15.jpeg'></td>
        <td><img src='./image/non_linear_10.jpeg'></td>
        <td><img src='./image/non_linear_5.jpeg'></td>
    </tr>
</table>
