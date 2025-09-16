function [stFull, params] = compute_stockwell(epochs, params)
    [nTime, nChannels, nTrials] = size(epochs);
    nFreq = length(params.psd.freq_range);
    stFull = zeros(nFreq, nTime, nChannels, nTrials);  % [freq x time x chan x trial]

    for iTrial = 1:nTrials
        for iChan = 1:nChannels
            stChannel = strans(epochs(:, iChan, iTrial), ...
                               params.psd.freq_range(1), ...
                               params.psd.freq_range(end), ...
                               params.fsamp, 1, 0, 1, 1, 1);  % Stockwell TFR
            stFull(:, :, iChan, iTrial) = stChannel;
        end
    end
end

function st = strans(timeseries,minfreq,maxfreq,samplingrate,freqsamplingrate,verbose,removeedge,analytic_signal,factor)  
% Returns the Stockwell Transform, STOutput, of the time-series 
% Code by R.G. Stockwell. 
% Reference is "Localization of the Complex Spectrum: The S Transform" 
% from IEEE Transactions on Signal Processing, vol. 44., number 4, 
% April 1996, pages 998-1001. 
% 
%-------Inputs Returned------------------------------------------------ 
%         - are all taken care of in the wrapper function above 
% 
%-------Outputs Returned------------------------------------------------ 
% 
%	ST    -a complex matrix containing the Stockwell transform. 
%			 The rows of STOutput are the frequencies and the 
%			 columns are the time values 
% 
% 
%----------------------------------------------------------------------- 
 
n=length(timeseries); 
original = timeseries; 
if removeedge 
    if verbose disp('Removing trend with polynomial fit'),end 
 	 ind = [0:n-1]'; 
    r = polyfit(ind,timeseries,2); 
    fit = polyval(r,ind) ; 
	 timeseries = timeseries - fit; 
    if verbose disp('Removing edges with 5% hanning taper'),end 
    sh_len = floor(length(timeseries)/10); 
    wn = hanning(sh_len); 
    if(sh_len==0) 
       sh_len=length(timeseries); 
       wn = 1&[1:sh_len]; 
    end 
    % make sure wn is a column vector, because timeseries is 
   if size(wn,2) > size(wn,1) 
      wn=wn';	 
   end 
    
   timeseries(1:floor(sh_len/2),1) = timeseries(1:floor(sh_len/2),1).*wn(1:floor(sh_len/2),1); 
	timeseries(length(timeseries)-floor(sh_len/2):n,1) = timeseries(length(timeseries)-floor(sh_len/2):n,1).*wn(sh_len-floor(sh_len/2):sh_len,1); 
   
end 
 
% If vector is real, do the analytic signal  
 
if analytic_signal 
   if verbose disp('Calculating analytic signal (using Hilbert transform)'),end 
   % this version of the hilbert transform is different than hilbert.m 
   %  This is correct! 
   ts_spe = fft(real(timeseries)); 
   h = [1; 2*ones(fix((n-1)/2),1); ones(1-rem(n,2),1); zeros(fix((n-1)/2),1)]; 
   ts_spe(:) = ts_spe.*h(:); 
   timeseries = ifft(ts_spe); 
end   
 
% Compute FFT's 
tic;vector_fft=fft(timeseries);tim_est=toc; 
vector_fft=[vector_fft,vector_fft]; 
tim_est = tim_est*ceil((maxfreq - minfreq+1)/freqsamplingrate)   ; 
if verbose disp(sprintf('Estimated time is %f',tim_est)),end 
 
% Preallocate the STOutput matrix 
st=zeros(ceil((maxfreq - minfreq+1)/freqsamplingrate),n); 
% Compute the mean 
% Compute S-transform value for 1 ... ceil(n/2+1)-1 frequency points 
if verbose disp('Calculating S transform...'),end 
if minfreq == 0 
   st(1,:) = mean(timeseries)*(1&[1:1:n]); 
else 
  	st(1,:)=ifft(vector_fft(minfreq+1:minfreq+n).*g_window(n,minfreq,factor)); 
end 
 
%the actual calculation of the ST 
% Start loop to increment the frequency point 
for banana=freqsamplingrate:freqsamplingrate:(maxfreq-minfreq) 
   st(banana/freqsamplingrate+1,:)=ifft(vector_fft(minfreq+banana+1:minfreq+banana+n).*g_window(n,minfreq+banana,factor)); 
end   % a fruit loop!   aaaaa ha ha ha ha ha ha ha ha ha ha 
 
end


