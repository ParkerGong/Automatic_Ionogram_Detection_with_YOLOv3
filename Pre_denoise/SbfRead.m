function SbfRead(sbffilename)
tic;
fid = fopen(sbffilename,'rb');%���ļ�
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

sizeO = size(imageO);
if sizeO(1)~=360
       close all;
       return;
end
%% ����X�ز���ȥ�����
se = strel('square', 4);
imageX = imdilate(imageX, se);
% ͼһ
figure,imagesc(imageO);colormap(1-gray);
  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The original ionogram')

    
%%% ȥ�� O�� X�� ���
%Ϊ��.raw ���ݷ�Χͳһ��0-32��������ѹ�� 256/8 = 32
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


%%%%��ֵΪ900*Ƶ�ʵ���Ƶ��ͼ%%%%
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
% ͼ��
figure,imagesc(imageO);colormap(1-gray);
  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21','23.5'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The original ionogram')
    
    %%%��ȡX�����ݣ���X���ٽ�Ƶ�ʸ�������ȥ��X�����ӣ����fx_I%%%%%%%%%%%%
[fx_I,imageX1] = SbfReadX(sbffilename);
fx_I = fx_I;
imageO = imageO - imageX1;
imageO(imageO<0) = 0;   
    
%%%%Ƶ��ͼԤ����ȥ��%%%%%%
imageO(imageO<1) = 0;
image_ori = imageO_II;
non_zero = length(find(imageO~=0));
noiseave = sum(sum(imageO))/non_zero;
for i = 1:freq_num_block
    for j = 1:900

        if imageO(j,i)< noiseave*1.1 %%%%%%��ֵ
            imageO(j,i) = 0;
        else
            imageO(j,i) = imageO(j,i)*8;
        end
    end
end
% ͼ��
figure,imagesc(imageO);colormap(1-gray);

  set(gca,'XTickLabel',{'3.5','6','8.5','11','13.5','16','18.5','21'})
    xlabel('Frequency (MHz)')
    set(gca,'YTickLabel',{'890','790','690','590','490','390','290','190','90'})
    ylabel('Virtual Height (km)')
    title('The ionogram after denoising')

% ͼ��
figure,imagesc(imageO);colormap(1-gray);
axis off
print('-djpeg', jpgfilename);
clc;clear;close all;
    
    
    