# Assignment 4  
## HDR imaging  
### Use dcraw to convert the RAQ  

먼저 net 형식의 파일을 dcraw.exe를 통해 tiff의 형식으로 바꾸었다.  
사용한 option은 아래와 같다.  
-w: use camera white balance  
-o 1: output colorspace(sRGB)  
-W: Don't automatically brighten the image  
-q 3: Set the interpolation quality high  
-4: linear 16-bit  
-T: write tiff instead of ppm  

아래는 사용 예시이다.  

<table>
    <tr>
        <th>convert nef file to tiff</th>
    </tr>
    <tr>
        <td><img src='./image/dcraw.jpg'></td>
    </tr>
</table>


### Linearize rendered images  

![Alt text](./image/weight.PNG)

위와 같은 weighting schemes를 이용하여 rendered image(jpg file)을 아래의 식을 통해 최적화하였다.  

![Alt text](./image/equation.PNG)

optimize한 결과는 아래의 그래프와 같다.  

<table>
    <tr>
        <th>g curve tent</th>
        <th>g curve uniform</th>
        <th>g curve gaussian</th>
    </tr>
    <tr>
        <td><img src='./image/tent.PNG'></td>
        <td><img src='./image/uniform.PNG'></td>
        <td><img src='./image/gaussian.PNG'></td>
    </tr>
</table>

```matlab
function output = poissonBlend(input, mask, target)

[height, width, channel] = size(input);
A = sparse(height*width, height*width);
b = zeros(height*width, channel);

im2var = zeros(height, width); 
im2var(1:height*width) = 1:height*width;

e = 0;

for h = 1:height
    for w = 1:width
        e = e + 1;
        if mask(h,w) == 1
            A(e, im2var(h, w)) = 4;
            A(e, im2var(h, w+1)) = -1;
            A(e, im2var(h, w-1)) = -1;
            A(e, im2var(h+1, w)) = -1;
            A(e, im2var(h-1, w)) = -1;
            b(e, :) = 4 * input(h, w, :) - input(h, w+1, :) - input(h, w-1, :) - input(h+1, w, :) - input(h-1, w, :);
        else
            A(e, im2var(h, w)) = 1;
            b(e, :) = target(h, w, :);
        end
    end
end

output = A\b;
output = reshape(output, [height width channel]);
end
```

<table>
    <tr>
        <th>poisson blending image</th>
    </tr>
    <tr>
        <td><img src='./image/poisson_blending.png'></td>
    </tr>
</table>

### Blending with mixed gradients  

이 부분은 source와 target의 gradient의 절대값 중 더 큰 값을 gradient로 사용하는 mixed gradients이다.  
모두 총 4가지 방향의 gradient를 계산하기 때문에,  
각각의 방향에서의 gradient 절대값들이 source와 target 중 더 큰 값을 적용해주어야 한다.  
그렇기 때문에 4방향의 gradient를 모두 비교하여 그 중 큰 값을 사용하도록 하였다.
```matlab
function output = mixedBlend(input, mask, target)

[height, width, channel] = size(input);
A = sparse(height*width, height*width);
b = zeros(height*width, channel);

im2var = zeros(height, width); 
im2var(1:height*width) = 1:height*width;

e = 0;

for h = 1:height
    for w = 1:width
        e = e + 1;
        if mask(h,w) == 1
            A(e, im2var(h, w)) = 4; 
            A(e, im2var(h, w+1)) = -1;
            A(e, im2var(h, w-1)) = -1;
            A(e, im2var(h+1, w)) = -1;
            A(e, im2var(h-1, w)) = -1;
            
            if abs(input(h, w, :) - input(h, w+1, :)) > abs(target(h, w, :) - target(h, w+1, :))
                width_plus_gradient = input(h, w, :) - input(h, w+1, :);
            else
                width_plus_gradient = target(h, w, :) - target(h, w+1, :);
            end
            
            if abs(input(h, w, :) - input(h, w-1, :)) > abs(target(h, w, :) - target(h, w-1, :))
                width_minus_gradient = input(h, w, :) - input(h, w-1, :);
            else
                width_minus_gradient = target(h, w, :) - target(h, w-1, :);
            end
            
            if abs(input(h, w, :) - input(h+1, w, :)) > abs(target(h, w, :) - target(h+1, w, :))
                height_plus_gradient = input(h, w, :) - input(h+1, w, :);
            else
                height_plus_gradient = target(h, w, :) - target(h+1, w, :);
            end
            
            if abs(input(h, w, :) - input(h-1, w, :)) > abs(target(h, w, :) - target(h-1, w, :))
                height_minus_gradient = input(h, w, :) - input(h-1, w, :);
            else
                height_minus_gradient = target(h, w, :) - target(h-1, w, :);
            end
            
             b(e, :) = width_plus_gradient + width_minus_gradient + height_plus_gradient + height_minus_gradient;
            
        else
            A(e, im2var(h, w)) = 1;
            b(e, :) = target(h, w, :);
        end
    end
end

output = A\b;
output = reshape(output, [height width channel]);
end
```

<table>
    <tr>
        <th>mixed gradients blending image</th>
    </tr>
    <tr>
        <td><img src='./image/mixed_blending.png'></td>
    </tr>
</table>

### Your own examples  

마지막은 다른 사진들을 가지고 직접 poisson blending과 mixed blending을 적용해보았다.  
아래의 이미지들과 같이 나타나는 것을 확인할 수 있었다.  
위의 poisson blending과 mixed blending의 결과의 차이와 같이  
poisson blending은 crop해서 붙여넣는 이미지의 모습이 어느정도 반영되는 것이 보이지만  
mixed blending의 경우에는 붙여넣는 이미지의 모습에 target의 무늬 등이 크게 반영이 된 것을 확인할 수 있다.

<table>
    <tr>
        <th>poisson blending</th>
        <th>mixed blending</th>
    </tr>
    <tr>
        <td><img src='./image/own_poisson.png'></td>
        <td><img src='./image/own_mixed.png'></td>
    </tr>
</table>
