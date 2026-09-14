function X = gedai_read(src, a, b, cls)
%GEDAI_READ  Samples a..b of a reflect-padded GEDAI source, channels x samples, in class cls.
%
%   X = gedai_read(src, a, b, cls)
%
%   Samples 1..src.P are the signal; samples beyond P are the reflect padding
%   GEDAI_per_band appends to reach a whole number of epochs, i.e. sample P+i
%   is signal sample P - mod(i-1, P) (the padding tiles when it is longer than
%   the signal). The result is cast to cls after it is formed, which is the
%   order the full-array code used: cast the band, then pad and slice it.
%
%   See also gedai_source.

if a > b
    X = zeros(src.N, 0, cls);
    return
end

P = src.P;
if b <= P
    X = local_signal(src, a, b);
elseif a > P
    X = local_padding(src, a - P, b - P);
else
    X = [local_signal(src, a, P), local_padding(src, 1, b - P)];
end
if ~isa(X, cls)
    X = cast(X, cls);
end
end

% =====================================================================
function X = local_padding(src, i0, i1)
% Padding samples i0..i1 read backwards from the end of the signal. Split into
% runs that each stay inside one reflection of the signal.
P = src.P;
X = zeros(src.N, i1 - i0 + 1, src.inputClass);
i = i0;
while i <= i1
    tileEnd = min(i1, ceil(i / P) * P);      % last padding index in this tile
    hi = P - mod(i - 1, P);                  % signal index of padding index i
    lo = P - mod(tileEnd - 1, P);
    X(:, i - i0 + 1 : tileEnd - i0 + 1) = fliplr(local_signal(src, lo, hi));
    i = tileEnd + 1;
end
end

% =====================================================================
function X = local_signal(src, a, b)
switch src.kind
    case 'matrix'
        X = src.data(:, a:b);
    case 'matrix_tc'
        X = src.data(a:b, :).';
    case 'wavelet'
        X = local_wavelet_band(src.data, src.band, src.level, a, b).';
    otherwise
        error('gedai_read:kind', 'Unknown source kind ''%s''.', src.kind);
end
end

% =====================================================================
function R = local_wavelet_band(A, f, L, a, b)
%LOCAL_WAVELET_BAND  Samples a..b of MODWT band f, samples x channels.
%   A is the level f-1 approximation, samples x channels. modwt_single_band
%   works on the whole record with circshift. Written per sample, its detail
%   band f and the synthesis back to level 0 are
%       D(t) = (A(t - s) - A(t)) * k,                    s = 2^(f-1)
%       R(t) = 0.5 * k * (D(t + s) - D(t))
%       R(t) = 0.5 * k * (R(t) + R(t + 2^(j-1))),        j = f-1 .. 1
%   with indices taken circularly, so samples a..b depend only on A over
%   [a - s, b + 2^f - 1]. That window is gathered once (circularly) and the same
%   expressions are applied to shifted slices of it. For the approximation band
%   (f = L+1) A is the level-L approximation and only the synthesis runs.
P = size(A, 1);
k = 1 / sqrt(2);
if f <= L
    s = 2^(f-1);
    W = A(mod((a - s : b + 2^f - 1) - 1, P) + 1, :);
    R = (W(1:end-s, :) - W(1+s:end, :)) * k;              % D over a .. b+2^f-1
    clear W
    R = 0.5 * k * (R(1+s:end, :) - R(1:end-s, :));        % over a .. b+2^(f-1)-1
    for j = f-1:-1:1
        t = 2^(j-1);
        R = 0.5 * k * (R(1:end-t, :) + R(1+t:end, :));
    end
else
    R = A(mod((a : b + 2^L - 1) - 1, P) + 1, :);
    for j = L:-1:1
        t = 2^(j-1);
        R = 0.5 * k * (R(1:end-t, :) + R(1+t:end, :));
    end
end
end
