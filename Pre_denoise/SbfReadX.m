function [fx_I,imageX1] = SbfReadX(sbffilename)
fid = fopen(sbffilename,'rb');%
jpgfilename = sbffilename;
strlen = length(sbffilename);
jpgfilename(strlen-2 : strlen) = 'jpg';
if (fid < 0)
       disp('Can not Open The File!');
end
fseek(fid, 0, 1);
filelen = ftell(fid);  %�����ļ����� �ֽ� ��λ
fseek(fid, 0, -1);

freq_num_block = filelen / 4096;   %����Ƶ�ʵ� block ��С 4096�ֽ�
filestream = fread(fid,filelen, 'uint8');
for i = 1 : freq_num_block
    imageO(:, i) = filestream((i-1)*4096 + 67 : (i-1)*4096 + 66 + 360);   % ���ݸ�ʽ ÿ��Ƶ�� 4096�ֽ�
    imageX(:, i) = filestream((i-1)*4096 + 571 : (i-1)*4096 + 570 + 360);%% 60�ֽ���ͷ + 6�ֽ� O��ͷ + 360�ֽ�O������  + 144�ֽ���� + 6�ֽ� X��ͷ + 360�ֽ�X������ + 144��� 
end
imageO(:,1:2) = 0; % ǰ�������ݲ�����
imageX(:,1:2) = 0;

imageO = flipud(imageO);
imageX = flipud(imageX);
%% ���X���ٽ�Ƶ�ʣ����д��f_I���㷨��O����ͬ %%%%%%%%
se = strel('square', 4);
imageO = imdilate(imageO, se);

%Ϊ��.raw ���ݷ�Χͳһ��0-32��������ѹ�� 256/8 = 32
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


%%%%Ƶ��ͼԤ����ȥ��%%%%%%
imageX(imageX<1) = 0;


%%%%%%ȥ��E��켣%%%%%
imageX(830:900,:)=0;
imageX(1:freq_num_block,:) = 0;
imageX(:,1:20) = 0;
%%%%%Ƶ��ͼ��ֵ��%%%%%%%%%
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

%%%����ͶӰ��X�����ͨ�����С���������������Ϊ��Ч�ز����%%%%%

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

%% ��Ƶ�ʻ��֣�ȡ��F��켣����Ƶ����ͶӰ������ٽ�Ƶ��  %%%%

if length(area)>3
    area(area<mean(area)*0.4)=0;%threshold_a
end
if length(area1)>3
    area1(area1<mean(area1)*0.4)=0;%threshold_a
end
max_area = max(area);
max_nn = find(max_area == area);
for i = max_nn+8:length(area) %%%%2017.4.14 8��Ϊ3
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



