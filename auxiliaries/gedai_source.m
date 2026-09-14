function src = gedai_source(kind, data, band, level)
%GEDAI_SOURCE  Read-only view of the signal that one GEDAI band pass runs on.
%
%   src = gedai_source('matrix', X)                 X is channels x samples
%   src = gedai_source('matrix_tc', X)              X is samples x channels
%   src = gedai_source('wavelet', A, band, level)   A is samples x channels
%
%   A band pass used to receive its signal as a full channels x samples matrix
%   and then make further full-size copies of it: a cast to the working
%   precision, a reflect-padded copy, a half-epoch-shifted copy for the second
%   epoch grid, and, in GEDAI.m, the MODWT reconstruction of the band itself.
%   For a whole night at 256 channels each of those is several GB. A source
%   replaces them with reads of sample ranges (gedai_read), and every read
%   returns exactly the values the matching slice of those copies held, as a
%   channels x samples matrix.
%
%   'matrix'     the signal itself, channels x samples (single or double).
%
%   'matrix_tc'  the signal itself, samples x channels. MATLAB stores arrays
%                column by column, so in this orientation one channel is one
%                contiguous column: per-channel work on a whole-night array
%                stays cache-friendly, and time slices are still cheap. Reads
%                transpose the slice.
%
%   'wavelet'    A is the Haar MODWT approximation at level band-1 of some
%                signal X0 (samples x channels, double; A is X0 itself for band
%                1) - the orientation modwt_single_band works in - and the
%                source is multiresolution band 'band' of a 'level'-level
%                decomposition of X0, i.e. modwt_single_band(X0, 'haar', level,
%                band). Bands 1..level are detail bands; band level+1 is the
%                approximation band, for which A must be the level-'level'
%                approximation. A read reconstructs only the requested samples,
%                from a window of A just wide enough to cover the synthesis
%                filters, with the same circular boundary rule as circshift;
%                gedai_wavelet_band reconstructs the whole band at once. Either
%                way the arithmetic per sample is modwt_single_band's, so the
%                values are bit-identical.
%
%   Fields
%     kind        'matrix', 'matrix_tc' or 'wavelet'
%     data        the array read from (shared with the caller, never modified)
%     N, P        channels and samples of the signal (unpadded)
%     band, level wavelet band and decomposition level ('wavelet' only)
%     inputClass  class the signal has as a matrix. A 'wavelet' band is
%                 reconstructed in double, as modwt_single_band does on double
%                 data. GEDAI.m's single-precision retry sets this to 'single',
%                 which is what passing single(band) used to mean.
%
%   The band engine decides the working precision and the reflect-padding
%   length; both are arguments of gedai_read rather than fields here.
%
%   See also gedai_read, gedai_wavelet_band, gedai_band_engine, modwt_single_band.

if nargin < 3, band = []; end
if nargin < 4, level = []; end

switch kind
    case 'matrix'
        if ~ismatrix(data)
            error('gedai_source:notMatrix', 'Input EEG data must be a 2D matrix (channels x samples).');
        end
        N = size(data, 1); P = size(data, 2);
    case 'matrix_tc'
        if ~ismatrix(data)
            error('gedai_source:notMatrix', 'Input EEG data must be a 2D matrix (samples x channels).');
        end
        N = size(data, 2); P = size(data, 1);
    case 'wavelet'
        if ~isa(data, 'double')
            error('gedai_source:approxClass', 'The wavelet approximation must be double.');
        end
        if isempty(band) || isempty(level) || band < 1 || band > level + 1
            error('gedai_source:band', 'band must lie in 1..level+1.');
        end
        N = size(data, 2); P = size(data, 1);
    otherwise
        error('gedai_source:kind', 'Unknown source kind ''%s''.', kind);
end

src = struct('kind', kind, 'data', data, 'N', N, 'P', P, ...
             'band', band, 'level', level, 'inputClass', class(data));
end
