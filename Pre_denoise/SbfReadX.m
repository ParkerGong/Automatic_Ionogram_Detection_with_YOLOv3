function [fx_I,imageX1] = SbfReadX(sbffilename)
fid = fopen(sbffilename,'rb');%
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
%% 获得X波临界频率，结果写入f_I，算法与O波相同 %%%%%%%%
se = strel('square', 4);
imageO = imdilate(imageO, se);

%为和.raw 数据范围统一【0-32】将数据压缩 256/8 = 32
imageO = imageO ./ 8;
imageX = imageX ./ 8;



for i=1:freq_num_block
        ii=1:360;
        rr=1:0.2:360;
        imageX_I(5:1800,i) = interp1(ii,imageX(:,i),rr);
        imageX_I(1:4,i) = imageX(1,i);
     
end
for i =1:900
   imageX_II(i,:) = imageX_I(i*2,:);
end
imageX = [];
imageX = imageX_II;


%%%%频高图预处理，去噪%%%%%%
imageX(imageX<1) = 0;


%%%%%%去除E层轨迹%%%%%
imageX(830:900,:)=0;
imageX(1:freq_num_block,:) = 0;
imageX(:,1:20) = 0;
%%%%%频高图二值化%%%%%%%%%
binary_data=im2bw(imageX);

X = sum(binary_data);%binary_data

  
if sum(X)==0
       fx_I = 0;
       imageX1 = zeros(900,400);
       disp('There is no trace found!');
       close all;
       return;
end

nonz = find(X==0);
cnt=1;
for i = 1: length(nonz)-1
    if X(nonz(i)+1)~=0
        ns1(cnt) = nonz(i);
        cnt = cnt+1;
    end
end

%%%计算投影在X轴的连通区域大小，保留面积最大的作为有效回波面积%%%%%

for i = 1:length(ns1)

    if i~=length(ns1)
        area1(i) =  sum(X(ns1(i):ns1(i+1)));
    else
        area1(i) = sum(X(ns1(i):end));
    end
end

cnt=1;
for i = length(nonz):-1:2
    if X(nonz(i)-1)~=0
        ns(cnt) = nonz(i);
        cnt = cnt+1;
    end
end

ns = fliplr(ns);
for i = 1:length(ns)

    if i~=1
        area(i) =  sum(X(ns(i-1):ns(i)));
    else
        area(i) = sum(X(1:ns(i)));
    end
end

%% 按频率划分，取出F层轨迹，向频率轴投影，获得临界频率  %%%%

if length(area)>3
    area(area<mean(area)*0.4)=0;%threshold_a
end
if length(area1)>3
    area1(area1<mean(area1)*0.4)=0;%threshold_a
end
max_area = max(area);
max_nn = find(max_area == area);
for i = max_nn+8:length(area) %%%%2017.4.14 8改为3
    if area(i)<80;
        area(i)=0;
    end
end
max_n = max(find(area~=0));
min_n = min(find(area1~=0));


f_max = ns(max_n);

f_min = ns1(min_n);


f_max = f_max+5; 
f_min = f_min;


imageX(:,f_max:freq_num_block)=0;   
imageX(:,1:f_min) = 0;
X(1:f_min)=0;X(f_max:freq_num_block)=0;

X_non = find(X~=0);
if length(X_non)>1
    for i=1:length(X_non)-1
        diff_X(i) = X_non(i+1)-X_non(i);
    end
    diff_Xm = find(diff_X>40);
    %%%%2017.4.14
    if ~isempty(diff_Xm)
        if diff_Xm(1)==1
        diff_Xm(1)=[];
        end
    end
    if ~isempty(diff_Xm)
        X(X_non(diff_Xm+1):end)=0;
        imageX(:,X_non(diff_Xm+1):end)=0;  
    end
end

X1 = sum(imageX);
fx_I = max(find(X1~=0));
imageX1 = zeros(900,freq_num_block);
if fx_I>30&&fx_I<380
    imageX1(:,fx_I-30:fx_I+10) = imageX (:,fx_I-30:fx_I+10);
end


fclose(fid);



