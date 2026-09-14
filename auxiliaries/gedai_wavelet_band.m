function B = gedai_wavelet_band(A, f, L, cls)
%GEDAI_WAVELET_BAND  MODWT band f of a whole recording, from its level f-1 approximation.
%
%   B = gedai_wavelet_band(A, f, L, cls)
%
%   A is the Haar MODWT approximation at level f-1 (samples x channels, double);
%   B is multiresolution band f of an L-level decomposition, samples x channels,
%   in class cls. For f <= L that is a detail band; for f = L+1 (the
%   approximation band) A must be the level-L approximation.
%
%   modwt_single_band's own detail and synthesis steps (circshift along time),
%   applied to a few channels at a time and cast into B as each piece is done.
%   A channel's values never depend on another channel, so B is sample for
%   sample what modwt_single_band returns for the whole matrix (cast to cls),
%   while only B and the working copies of those few channels are held.
%   modwt_single_band additionally rebuilds the level f-1 approximation from
%   level 0 on every call; the caller keeps it instead.
%
%   See also gedai_source, modwt_single_band.

[P, N] = size(A);
B = zeros(P, N, cls);
k = 1 / sqrt(2);
channels_per_piece = max(1, floor(256 * 2^20 / (8 * P)));
for c0 = 1:channels_per_piece:N
    chans = c0:min(N, c0 + channels_per_piece - 1);
    R = A(:, chans);
    if f <= L
        s = 2^(f-1);
        R = (circshift(R, s, 1) - R) * k;
        R = 0.5 * k * (circshift(R, -s, 1) - R);
        for j = f-1:-1:1
            R = 0.5 * k * (R + circshift(R, -2^(j-1), 1));
        end
    else
        for j = L:-1:1
            R = 0.5 * k * (R + circshift(R, -2^(j-1), 1));
        end
    end
    B(:, chans) = R;
end
end
