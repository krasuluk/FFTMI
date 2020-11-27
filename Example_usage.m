HDR = hdrread('C1.hdr');
% Note that the image comes from the TMQID database, where the values are
% already scaled to physical luminance (i.e. every pixel value represents 
% the exact number of nits that were emitted). If this is not the case, 
% a display model needs to be used to scale the values.

LDR = imread('C1_Drago1.png');

[FFTMI_score, SS2, FN, FSITM_r, FSITM_g, FSITM_b] = FFTMI(HDR, LDR);

