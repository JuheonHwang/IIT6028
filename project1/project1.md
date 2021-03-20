# Assignment 1  
## Implement a basic image processing pipeline  
### Initials

```matlab
clc; clear all;
RAW = imread('banana_slug.tiff');

sz = size(RAW);
className = class(RAW);
db = double(RAW);
```
