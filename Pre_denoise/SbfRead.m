function SbfRead(sbffilename)
tic;
fid = fopen(sbffilename,'rb');%打开文件
jpgfilename = sbffilename;
strlen = length(sbffilename);
jpgfilename(strlen-2 : strlen) = 'jpg';

if (fid < 0)
       disp('Can not Open The File!');
end
fseek(fid, 0, 1);
filelen = ftell(fid);  %计算文件长度 字节 单位
fseek(fid, 0, -1);

freq_num_block = filelen / 4096;   %计算频率点 block 大小 4096字节
filestream = fread(fid,filelen, 'uint8');
for i = 1 : freq_num_block
    imageO(:, i) = filestream((i-1)*4096 + 67 : (i-1)*4096 + 66 + 360);   % 数据格式 每个频点 4096字节
    imageX(:, i) = filestream((i-1)*4096 + 571 : (i-1)*4096 + 570 + 360);%% 60字节组头 + 6字节 O波头 + 360字节O波数据  + 144字节填充 + 6字节 X波头 + 360字节X波数据 + 144填充 
end
imageO(:,1:2) = 0; % 前两列数据不能用
imageX(:,1:2) = 0;

imageO = flipud(imageO);
imageX = flipud(imageX);

sizeO = size(imageO);
if sizeO(1)~=360
       close all;
       return;
end
%% 膨胀X回波，去除混叠
se = strel('square', 4);
imageX = imdilate(imageX, se);
% 图一
figure,imagesc(imageO);colormap(1-gray);
  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The original ionogram')

    
%%% 去除 O波 X波 混叠
%为和.raw 数据范围统一【0-32】将数据压缩 256/8 = 32
imageO = imageO ./ 8;
imageX = imageX ./ 8;
for i = 1 : 360
    for j = 1 : freq_num_block
        if (imageO(i,j) > imageX(i,j) && imageO(i,j) >20)
            imageX(i,j) = 0;
        elseif (imageX(i,j) > imageO(i,j) && imageX(i,j) > 20)
            imageO(i,j) = 0;
        end
    end
end


%%%%插值为900*频率点数频高图%%%%
for i=1:freq_num_block
        ii=1:360;
        rr=1:0.2:360;
        imageO_I(5:1800,i) = interp1(ii,imageO(:,i),rr);
        imageO_I(1:4,i) = imageO(1,i);
     
end
for i =1:900
   imageO_II(i,:) = imageO_I(i*2,:);
end
imageO = [];
imageO = imageO_II;
% 图二
figure,imagesc(imageO);colormap(1-gray);
  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21','23.5'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The original ionogram')
    
    %%%读取X波数据，用X波临界频率附近数据去除X波混杂，获得fx_I%%%%%%%%%%%%
[fx_I,imageX1] = SbfReadX(sbffilename);
fx_I = fx_I;
imageO = imageO - imageX1;
imageO(imageO<0) = 0;   
    
%%%%频高图预处理，去噪%%%%%%
imageO(imageO<1) = 0;
image_ori = imageO_II;
non_zero = length(find(imageO~=0));
noiseave = sum(sum(imageO))/non_zero;
for i = 1:freq_num_block
    for j = 1:900

        if imageO(j,i)< noiseave*1.1 %%%%%%阈值
            imageO(j,i) = 0;
        else
            imageO(j,i) = imageO(j,i)*8;
        end
    end
end
% 图三
figure,imagesc(imageO);colormap(1-gray);

  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The ionogram after denoising')

% 图四
figure,imagesc(imageO);colormap(1-gray);
axis off
print('-djpeg', jpgfilename);
clc;clear;close all;
    
    
    