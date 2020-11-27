function [FFTMI_score, SS2, FN, FSITM_r, FSITM_g, FSITM_b] = FFTMI(HDR, LDR)
     % The funciton implements the objective quality metric described in the paper:
     %
     % L. Krasula, K. Fliegel and P. Le Callet, "FFTMI: Features Fusion for Natural 
     % Tone-Mapped Images Quality Evaluation," in IEEE Transactions on Multimedia, 
     % vol. 22, no. 8, pp. 2038-2047, Aug. 2020, doi: 10.1109/TMM.2019.2952256.
     %
     % When you use our method in your research, please, cite the above stated
     % paper.
     %
     % Copyright (c) 2020
     % Lukas Krasula <l.krasula@gmail.com>
    
     % Permission to use, copy, modify, and/or distribute this software for any
     % purpose with or without fee is hereby granted, provided that the above
     % copyright notice and this permission notice appear in all copies.
     %
     % THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
     % WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
     % MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
     % ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
     % WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
     % ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
     % OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
     %
     % This software also uses the code described in:
     % L. Krasula, K. Fliegel, P. Le Callet, M. Klíma, “Objective Evaluation 
     % of Naturalness, Contrast, and Colorfulness of Tone-Mapped Images,” 
     % Proc. SPIE 9217, Applications of Digital Image Processing XXXVII, 
     % doi:10.1117/12.2075270
     %
     % K. Ma, H. Yeganeh, K. Zeng, and Z. Wang, “High dynamic range image compression
     % by  optimizing  tone  mapped  image  quality  index, ”IEEETransactions on Image
     % Processing, vol. 24, no. 10, pp. 3086–3097, 2015
     %
     % H. Ziaei Nafchi, A. Shahkolaei, R. Farrahi Moghaddam, and M. Cheriet,
     % “FSITM:  A  feature  similarity  index  for  tone-mapped  images,
     % ”IEEE Signal Processing Letters, vol. 22, no. 8, pp. 1026–1029, 2015
     %
     %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % HDR - an HDR image (MxNx3) scaled to physical luminance (in nits)
     % LDR - a gamma-corrected 8-bit tone-mapped version of the HDR image
     %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % FFTMI_score - a score describing the quality of the LDR image 
    
SS2 = structural_similarity(HDR, LDR);
FN = feature_naturalness(LDR);
FSITM_r = FSITM(HDR, LDR, 1);
FSITM_g = FSITM(HDR, LDR, 2);
FSITM_b = FSITM(HDR, LDR, 3);

FFTMI_score = 0.2129 * SS2 + 0.0443 * FN + 1 * FSITM_r + 0.0621 * FSITM_g + 0.0931 * FSITM_b;

end



function [S, sMap] = structural_similarity(hdrI, ldrI, window)
% ========================================================================
% Tone Mapped image Quality Index (TMQI), Version 2.0
% Copyright(c) 2014 Kede Ma, Hojatollah Yeganeh, Kai Zeng and Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% This is an implementation of an objective image quality assessment model
% for tone mapped low dynamic range (LDR) images using their corresponding
% high dynamic range (HDR) images as references.
% 
% Please refer to the following papers and the website with suggested usage
%
% K. Ma et al., "High Dynamic Range Image Compression by Optimizing Tone
% Mapped Image Quality Index" to be submitted to IEEE Transactions on 
% Image Processing.
%
% H. Yeganeh and Z. Wang, "Objective Quality Assessment of Tone Mapped
% Images," IEEE Transactios on Image Processing, vol. 22, no. 2, pp. 657- 
% 667, Feb. 2013.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/tmqi/
%
% Kindly report any suggestions or corrections to k29ma@uwaterloo.ca,
% hojat.yeganeh@gmail.com, kzeng@uwaterloo.ca or zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) hdrI: the HDR image being used as reference.
%        (2) ldrI: the LDR image being compared (either color or
%                  grayscale image with its dynamic range equal to 255)
%        (3) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%
%Output: (1) Q: The TMQI-II score of the LDR image. 
%        (2) S: The structural fidelity score of the LDR test image.
%        (3) N: The statistical naturalness score of the LDR image.
%        (4) sMap: The structural fidelity map of the LDR image. 
%
%Basic Usage:
%   Given LDR test image and its corresponding HDR image, 
%
%   [Q, S, N, sMap] = TMQI2(hdrI, ldrI);
%
%Advanced Usage:
%   User defined parameters. For example
%   window = ones(8);
%   [Q, S, N, sMap] = TMQI2(hdrI, ldrI, window);
%
%========================================================================

if (nargin < 2 || nargin > 3)
   S = -Inf;
   sMap = -Inf;
   return;
end

[s1, s2, s3] = size(hdrI);

if (size(ldrI,1) ~= s1 || size(ldrI,2) ~= s2)
   S = -Inf;
   sMap = -Inf;
   return;
end

if (nargin == 2)
   if ((s1 < 11) || (s2 < 11))
       S = -Inf;
	   sMap = -Inf;
       disp('The image size is less than the window size.'); 
     return;
   end
   window = fspecial('gaussian', 11, 1.5);	%
end

if (nargin == 3)
   [H, W] = size(window);
   if ((H*W) < 4 || (H > s1) || (W > s2))
       S = -Inf;
      sMap = -Inf;
      return;
   end
end

%---------- default parameters -----
hdrTh = 0.06; % threshold for HDR images
ldrTh = 2.6303; % threshold for LDR images
C1 = 0.01; % constants to avoid instability
C2 = 10;
%-------------------------------------

window = window/sum(window(:));
hdrI = double(hdrI);
ldrI = double(ldrI);
if s3 == 3
    hdrL = 0.213 * hdrI(:,:,1) + 0.715 * hdrI(:,:,2) + 0.072 * hdrI(:,:,3); % extract luminance component
else
    hdrL = hdrI;
end
minL = min(hdrL(:));
maxL = max(hdrL(:));
hdrL = double( round( (2^32 - 1) / (maxL - minL) ) .* (hdrL - minL) ); % full contrast stretch 
if size(ldrI,3) == 3
    ldrL = 0.213 * ldrI(:,:,1) + 0.715 * ldrI(:,:,2) + 0.072 * ldrI(:,:,3); 
else
    ldrL = ldrI;
end

%================= Structural Fidelity Measure =========================
muH   = filter2(window, hdrL, 'valid');
muL   = filter2(window, ldrL, 'valid');
muH_sq = muH.*muH;
muL_sq = muL.*muL;
muH_muL = muH.*muL;
sigmaH_sq = filter2(window, hdrL.*hdrL, 'valid') - muH_sq;
sigmaL_sq = filter2(window, ldrL.*ldrL, 'valid') - muL_sq;
sigmaH = sqrt(max(0, sigmaH_sq));
sigmaL = sqrt(max(0, sigmaL_sq));
sigmaHL = filter2(window, hdrL.*ldrL, 'valid') - muH_muL;

sigmaHp = normcdf(sigmaH ./ muH, hdrTh, hdrTh/3); % sigma normalized by the mean
sigmaLp = normcdf(sigmaL, ldrTh, ldrTh/3);
%----------------------------------------------------
sMap = ( (( 2 * sigmaHp .* sigmaLp ) + C1 ) ./ ( ( sigmaHp .* sigmaHp ) + ( sigmaLp .* sigmaLp ) + C1 ) ) .* ( ( sigmaHL + C2) ./ ( sigmaH .* sigmaL + C2 ) );
S = nanmean(sMap(:)); 
end




function [FN,CQE1,GCF,M] = feature_naturalness(im)

CQE1 = cqe1_colorfulness(im);
GCF = fast_global_contrast(im);

if(max(im(:))>1)
    im2 = im./255;
    M = mean(im2(:));    
end

    FEA = CQE1.*GCF.*M;
    ra = raylpdf(0:0.001:2,0.27);
    FN = raylpdf(FEA,0.27)/max(ra);
end




function [CQE1,mu_alpha,mu_beta,s_alpha2,s_beta2] = cqe1_colorfulness(image)
image = double(image);

if(max(image(:))<=1)
    image = image.*255;
end

alpha = image(:,:,1) - image(:,:,2);
beta = 0.5*(image(:,:,1) + image(:,:,2)) - image(:,:,3);

mu_alpha = mean(alpha(:));
mu_beta = mean(beta(:));

s_alpha2 = mean( alpha(:).^2 - mu_alpha^2);
s_beta2 = mean( beta(:).^2 - mu_beta^2);

CQE1 = 0.02*log10(s_alpha2/(abs(mu_alpha)^0.2))*log10(s_beta2/(abs(mu_beta)^0.2));

if(CQE1 == Inf||CQE1 == -Inf||isnan(CQE1))
    CQE1 = 0;
end

end





function [GCF] = fast_global_contrast(image)

%% Extracting the luminance component
if (ndims(image) == 3)
   image = double(rgb2ycbcr(image));
   Y = image(:,:,1);
else
    Y = double(image);
end;
    
if(max(Y(:)) > 1)
    Y = Y./255;
end

%% Inicialization (according to the paper)
scales = [1 2 4 8 16 25 50 100 200];
l = Y.^2.2;

C = zeros(1,length(scales));
C_w = zeros(1,length(scales));
for i = 1:length(scales)
    L = [];
    l_scale = [];
    C_L = [];
    C_L2 = [];
    
    % Creating image on different scales
    if(scales(i)==1)
        l_scale = l;
    else
        COLS = mean(im2col(l,[scales(i) scales(i)],'distinct'));
        l_scale = reshape(COLS,[ceil(size(l,1)/scales(i)) ceil(size(l,2)/scales(i))]);
        COLS = [];
    end
    L = 100.*sqrt(l_scale);
    
    % padding for filtering purposes (not mirroring the border element but the one next to it)
    L2 = padarray(L,[2 2],'symmetric'); L2(2,:) = []; L2(:,2) = []; L2(end-1,:) = []; L2(:,end-1) = [];
    % matrix implementation of this: (abs(x(1,2)-x(2,2))+abs(x(2,1)-x(2,2))+abs(x(2,3)-x(2,2))+abs(x(3,2)-x(2,2)))/4;
    COLS = im2col(L2,[3 3],'sliding');
    COLS2 = mean(abs(COLS([2,4,6,8],:)-(ones(4,1)*COLS(5,:))));
    COLS = [];
    C_L = reshape(COLS2,[size(L,1) size(L,2)]);
    COLS2 = [];

    C_L(C_L == Inf) = 0;
    
    C(i) = mean(C_L(:));
    C_w(i) = C(i) * ((-0.406385 * i/length(scales) + 0.334573) * i/length(scales) + 0.0877526);    
end

GCF = sum(C_w);
end





function Q = FSITM (HDR, LDR, CH)

% Feature similarity index for tone mapped images (FSITM)

% By: Hossein Ziaei Nafchi, November 2014
% hossein.zi@synchromedia.ca
% Synchromedia Lab, ETS, Canada

% The code can be modified, rewritten and used without obtaining permission
% of the authors.

% Please refer to the following paper:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Reza Farrahi Moghaddam, Mohamed Cheriet, IEEE Signal Processing Letters, vol. 22, no. 8, pp. 1026-1029, 2015.

%%
% HDR: High dynamic range image
% LDR: Low dynamic range image
% CH = 1 --> Red channel, CH = 2 --> Green channel, CH = 3 --> Blue channel
% Q: Quality index

% Needs phasecong100 and Lowpassfilter functions

%%
[row, col, ~] = size(LDR);
NumPixels = row * col;

r = floor(NumPixels / (2 ^ 18));

if r > 1
    alpha = 1 - (1 / r);
else
    alpha = 0;
end


HDR_CH = HDR(:, :, CH);
LDR_CH = LDR(:, :, CH);

LogH = HDR_CH;
minNonzero = min(HDR_CH(HDR_CH ~= 0));
LogH(HDR_CH == 0) = minNonzero;
LogH = log(LogH);
LogH = im2uint8(mat2gray(LogH)); 


if alpha~=0
    HDR_CH = HDR(:, :, CH); 
    PhaseHDR_CH = phasecong100(HDR_CH, 2, 2, 8, 8);
    PhaseLDR_CH8 = phasecong100(LDR_CH, 2, 2, 8, 8);
else
    PhaseHDR_CH = 0;
    PhaseLDR_CH8 = 0;
end

PhaseLogH = phasecong100(LogH, 2, 2, 2, 2);
PhaseH = alpha * PhaseHDR_CH + (1 - alpha) * PhaseLogH; 



PhaseLDR_CH2 = phasecong100(LDR_CH, 2, 2, 2, 2);
PhaseL = alpha * PhaseLDR_CH8 + (1 - alpha) * PhaseLDR_CH2;

index = (PhaseL <= 0 & PhaseH <= 0) | (PhaseL > 0 & PhaseH > 0);
Q = sum(index(:)) / NumPixels;
end




% LOWPASSFILTER - Constructs a low-pass butterworth filter.
%
% usage: f = Lowpassfilter(sze, cutoff, n)
% 
% where: sze    is a two element vector specifying the size of filter 
%               to construct [rows cols].
%        cutoff is the cutoff frequency of the filter 0 - 0.5
%        n      is the order of the filter, the higher n is the sharper
%               the transition is. (n must be an integer >= 1).
%               Note that n is doubled so that it is always an even integer.
%
%                      1
%      f =    --------------------
%                              2n
%              1.0 + (w/cutoff)
%
% The frequency origin of the returned filter is at the corners.
%
% See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER
%

% Copyright (c) 1999 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% October 1999
% August  2005 - Fixed up frequency ranges for odd and even sized filters
%                (previous code was a bit approximate)

function f = Lowpassfilter(sze, cutoff, n)
    
    if cutoff < 0 | cutoff > 0.5
	error('cutoff frequency must be between 0 and 0.5');
    end
    
    if rem(n,1) ~= 0 | n < 1
	error('n must be an integer >= 1');
    end

    if length(sze) == 1
	rows = sze; cols = sze;
    else
	rows = sze(1); cols = sze(2);
    end

    % Set up X and Y matrices with ranges normalised to +/- 0.5
    % The following code adjusts things appropriately for odd and even values
    % of rows and columns.
    if mod(cols,2)
	xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
    else
	xrange = [-cols/2:(cols/2-1)]/cols;	
    end

    if mod(rows,2)
	yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
    else
	yrange = [-rows/2:(rows/2-1)]/rows;	
    end
    
    [x,y] = meshgrid(xrange, yrange);
    radius = sqrt(x.^2 + y.^2);        % A matrix with every pixel = radius relative to centre.
    f = ifftshift( 1.0 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   % The filter
end



function featType = phasecong100(varargin)


% Copyright (c) 1996-2010 Peter Kovesi Centre for Exploration Targeting The University of Western Australia peter.kovesi@uwa.edu.au


%% This function is optimized to generate one of the outputs of the 'phasecong3' function, please see the original function at:
%  http://www.csse.uwa.edu.au/~pk/research/matlabfns/
%%
%     im                       % Input image
%     nscale          = 2;     % Number of wavelet scales.    
%     norient         = 2;     % Number of filter orientations.
%     minWaveLength   = 7;     % Wavelength of smallest scale filter.    
%     mult            = 2;     % Scaling factor between successive filters.    
%     sigmaOnf        = 0.65;  % Ratio of the standard deviation of the
                               % Gaussian describing the log Gabor filter's
                               % transfer function in the frequency domain
                               % to the filter center frequency.


% Get arguments and/or default values    
    [im, nscale, norient, minWaveLength, mult, sigmaOnf] = checkargs(varargin(:));     

   
    
    
    
    [rows,cols] = size(im);
    imagefft = fft2(im);              % Fourier transform of image

    zero = zeros(rows,cols);
    EO = cell(nscale, norient);       % Array of convolution results.  
    

    EnergyV = zeros(rows,cols,3);     % Matrix for accumulating total energy
                                      % vector, used for feature orientation
                                      % and type calculation

    
    
    % Set up X and Y matrices with ranges normalised to +/- 0.5
    % The following code adjusts things appropriately for odd and even values
    % of rows and columns.
    if mod(cols,2)
        xrange = (-(cols-1)/2:(cols-1)/2)/(cols-1);
    else
        xrange = (-cols/2:(cols/2-1))/cols; 
    end
    
    if mod(rows,2)
        yrange = (-(rows-1)/2:(rows-1)/2)/(rows-1);
    else
        yrange = (-rows/2:(rows/2-1))/rows; 
    end
    
    [x,y] = meshgrid(xrange, yrange);
    
    radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
    theta = atan2(-y,x);              % Matrix values contain polar angle.
                                      % (note -ve y is used to give +ve
                                      % anti-clockwise angles)
                                  
    radius = ifftshift(radius);       % Quadrant shift radius and theta so that filters
    theta  = ifftshift(theta);        % are constructed with 0 frequency at the corners.
    radius(1,1) = 1;                  % Get rid of the 0 radius value at the 0
                                      % frequency point (now at top-left corner)
                                      % so that taking the log of the radius will 
                                      % not cause trouble.
    sintheta = sin(theta);
    costheta = cos(theta);
    clear x; clear y; clear theta;    % save a little memory
    
    % Filters are constructed in terms of two components.
    % 1) The radial component, which controls the frequency band that the filter
    %    responds to
    % 2) The angular component, which controls the orientation that the filter
    %    responds to.
    % The two components are multiplied together to construct the overall filter.
    
    % Construct the radial filter components...
    % First construct a low-pass filter that is as large as possible, yet falls
    % away to zero at the boundaries.  All log Gabor filters are multiplied by
    % this to ensure no extra frequencies at the 'corners' of the FFT are
    % incorporated as this seems to upset the normalisation process when
    % calculating phase congrunecy.
    lp = Lowpassfilter([rows,cols],.45,15);   % Radius .45, 'sharpness' 15

    logGabor = cell(1,nscale);

    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1);
        fo = 1.0/wavelength;                  % Centre frequency of filter.
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
        logGabor{s} = logGabor{s}.*lp;        % Apply low-pass filter
        logGabor{s}(1,1) = 0;                 % Set the value at the 0 frequency point of the filter
                                              % back to zero (undo the radius fudge).
    end
    
    %% The main loop...
    
    for o = 1:norient                    % For each orientation...
        % Construct the angular filter spread function
        angl = (o-1)*pi/norient;         % Filter angle.
        % For each point in the filter matrix calculate the angular distance from
        % the specified filter orientation.  To overcome the angular wrap-around
        % problem sine difference and cosine difference values are first computed
        % and then the atan2 function is used to determine angular distance.
        ds = sintheta * cos(angl) - costheta * sin(angl);    % Difference in sine.
        dc = costheta * cos(angl) + sintheta * sin(angl);    % Difference in cosine.
        dtheta = abs(atan2(ds,dc));                          % Absolute angular distance.
        % Scale theta so that cosine spread function has the right wavelength and clamp to pi    
        dtheta = min(dtheta*norient/2,pi);
        % The spread function is cos(dtheta) between -pi and pi.  We add 1,
        % and then divide by 2 so that the value ranges 0-1
        spread = (cos(dtheta)+1)/2;        
        
        sumE_ThisOrient   = zero;          % Initialize accumulator matrices.
        sumO_ThisOrient   = zero;       
             

        for s = 1:nscale,                  % For each scale...
            filter = logGabor{s} .* spread;      % Multiply radial and angular
                                                 % components to get the filter. 
                                                 
            % Convolve image with even and odd filters returning the result in EO
            EO{s,o} = ifft2(imagefft .* filter);      

            
            sumE_ThisOrient = sumE_ThisOrient + real(EO{s,o}); % Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + imag(EO{s,o}); % Sum of odd filter convolution results.
        
        end                                       % ... and process the next scale

        % Accumulate total 3D energy vector data, this will be used to
        % determine overall feature orientation and feature phase/type
        EnergyV(:,:,1) = EnergyV(:,:,1) + sumE_ThisOrient;
        EnergyV(:,:,2) = EnergyV(:,:,2) + cos(angl)*sumO_ThisOrient;
        EnergyV(:,:,3) = EnergyV(:,:,3) + sin(angl)*sumO_ThisOrient;
       
    end  % For each orientation

    
    
    % feature phase/type computation
    OddV = sqrt(EnergyV(:,:,2).^2 + EnergyV(:,:,3).^2);
    featType = atan2(EnergyV(:,:,1), OddV);  % Feature phase  pi/2 <-> white line,
                                             % 0 <-> step, -pi/2 <-> black line
end

%%------------------------------------------------------------------
% CHECKARGS
%
% Function to process the arguments that have been supplied, assign
% default values as needed and perform basic checks.
    
function [im, nscale, norient, minWaveLength, mult, sigmaOnf] = checkargs(arg)

    nargs = length(arg);
    
    if nargs < 1
        error('No image supplied as an argument');
    end    
    
    % Set up default values for all arguments and then overwrite them
    % with with any new values that may be supplied
    im              = [];
    nscale          = 2;     % Number of wavelet scales.    
    norient         = 2;     % Number of filter orientations.
    minWaveLength   = 7;     % Wavelength of smallest scale filter.    
    mult            = 2;     % Scaling factor between successive filters.    
    sigmaOnf        = 0.65;  % Ratio of the standard deviation of the
                             % Gaussian describing the log Gabor filter's
                             % transfer function in the frequency domain
                             % to the filter center frequency.    
                               
    
    % Allowed argument reading states
    allnumeric   = 1;       % Numeric argument values in predefined order
    keywordvalue = 2;       % Arguments in the form of string keyword
                            % followed by numeric value
    readstate = allnumeric; % Start in the allnumeric state
    
    if readstate == allnumeric
        for n = 1:nargs
            if isa(arg{n},'char')
                readstate = keywordvalue;
                break;
            else
                if     n == 1, im            = arg{n}; 
                elseif n == 2, nscale        = arg{n};              
                elseif n == 3, norient       = arg{n};
                elseif n == 4, minWaveLength = arg{n};
                elseif n == 5, mult          = arg{n};
                elseif n == 6, sigmaOnf      = arg{n};
                end
            end
        end
    end

    % Code to handle parameter name - value pairs
    if readstate == keywordvalue
        while n < nargs
            
            if ~isa(arg{n},'char') || ~isa(arg{n+1}, 'double')
                error('There should be a parameter name - value pair');
            end
            
            if     strncmpi(arg{n},'im'      ,2), im =        arg{n+1};
            elseif strncmpi(arg{n},'nscale'  ,2), nscale =    arg{n+1};
            elseif strncmpi(arg{n},'norient' ,4), norient =   arg{n+1};
            elseif strncmpi(arg{n},'minWaveLength',2), minWaveLength = arg{n+1};
            elseif strncmpi(arg{n},'mult'    ,2), mult =      arg{n+1};
            elseif strncmpi(arg{n},'sigmaOnf',2), sigmaOnf =  arg{n+1};
            else   error('Unrecognised parameter name');
            end

            n = n+2;
            if n == nargs
                error('Unmatched parameter name - value pair');
            end
            
        end
    end
    
    if isempty(im)
        error('No image argument supplied');
    end

    if ~isa(im, 'double')
        im = double(im);
    end
    
    if nscale < 1
        error('nscale must be an integer >= 1');
    end
    
    if norient < 1 
        error('norient must be an integer >= 1');
    end    

    if minWaveLength < 2
        error('It makes little sense to have a wavelength < 2');
    end          

    
    
end