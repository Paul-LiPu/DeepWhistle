function [peak] = consolidate_peaks(peak, smoothed_dB, min_bin_gap)
    
    peak_dist = diff(peak);
    too_close_idx = find(peak_dist < min_bin_gap);

    while(~isempty(too_close_idx))
        fprintf('Consolidating peaks\n');
        to_close_vals = smoothed_dB(too_close_idx);
        [~, maxidx] = max(to_close_vals);
        max_freq_idx = too_close_idx(maxidx);
        
        j = max_freq_idx;
        if peak(j) > size(smoothed_dB)
            delete_idx = max_freq_idx;
            peak(delete_idx) = [];
            smoothed_dB(delete_idx) = [];
            
            peak_dist = diff(peak);
            too_close_idx = find(peak_dist < 2);
            continue
        end
%         
        j = j+1;
        if peak(j) > length(smoothed_dB)
            delete_idx = max_freq_idx +1;
            peak(delete_idx) = [];
            smoothed_dB(delete_idx) = [];
            
            peak_dist = diff(peak);
            too_close_idx = find(peak_dist < 2);
            continue
        end
                
        [~, max_offset] = ...
               min(smoothed_dB(peak([max_freq_idx; max_freq_idx+1])));
        delete_idx = max_freq_idx - 1 + max_offset;
           
        peak(delete_idx) = [];
        smoothed_dB(delete_idx) = [];

        peak_dist = diff(peak);
        too_close_idx = find(peak_dist < 2);
    end
end




