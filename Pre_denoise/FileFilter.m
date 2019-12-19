 %clc; clear; close all;


sbffilestruc = dir('*.sbf');
N = numel(sbffilestruc);

for i = 1: N
    filename = sbffilestruc(i).name;
    SbfRead(filename);
end
SbfRead('Z_SWGO_I_59441_20161214184502_O_INSD_DIS.sbf');