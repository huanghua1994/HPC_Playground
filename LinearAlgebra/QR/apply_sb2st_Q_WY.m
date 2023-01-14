function Z = apply_sb2st_Q_WY(VT, bs, nv, C)
% Apply the Q matrix from sb2st to C (calculate Q * C)
% Based on DOI:10.1016/j.parco.2011.05.002 Figure 2 right part and 
% DOI:10.1109/SHPCC.1994.296622 Section 3
% Input parameters:
%   VT : Size n * n, tril(VT, -2) stores the Householder vectors in sb2st
%   bs : Semi-bandwidth of the AB matrix entering sb2st, == length of each 
%        Householder vectors in sb2st
%   nv : Number of Householder vectors to combine
%   C  : Size n * k, matrix to be applied with Q
% Output parameter:
%   Z : Size n * k, == Q * C
    n = size(VT, 1);
    l_num = zeros(n-2, 1);
    for k = 1 : n-2
        j_list = [k, k+1 : bs : n-bs-1];
        l_num(k) = length(j_list);
    end
    for l = 1 : l_num(1)
        for k1 = n-2 : -1 : 1
            if (l_num(k1) >= l), break; end
        end
        for ke = k1 : -nv : 1
            % Combine transforms (ke : -1 : ks, l)
            ks = max(1, ke - nv + 1);
            curr_nv = ke - ks + 1;
            Y = zeros(bs + curr_nv - 1, curr_nv);
            min_r1 = 1e100;
            max_r2 = 0;
            for k = ke : -1 : ks
                % Calculate j, r1, r2 from (k, l) and extract corresponding v
                if (l == 1)
                    j = k;
                    r1 = j + 1;
                else
                    j = k + 1;
                    if (l > 2), j = j + (l - 2) * bs; end
                    r1 = j + bs;
                end
                r2 = min(r1 + bs - 1, n);
                v = VT(r1 : r2, k);
                if (l == 1), v(1) = 1; end
                % Record min_r1, max_r2; put v in Y
                min_r1 = min(min_r1, r1);
                max_r2 = max(max_r2, r2);
                Y_r1 = k - ks + 1;
                Y_r2 = Y_r1 + (r2 - r1);
                Y_col = ke - k + 1;
                Y(Y_r1 : Y_r2, Y_col) = v;
            end
            % length(v) might be smaller than bs, truncate empty rows in Y
            Y = Y(1 : max_r2-min_r1+1, :);
            W = gen_W_from_Y(Y);
            % Q' * C(min_r1 : max_r2, :) = (I - Y * W') * C(min_r1 : max_r2, :)
            % Why we are using Q' * C instead of Q * C here, but Q * C in apply_sy2sb_Q_WY?
            WTC = W' * C(min_r1 : max_r2, :);
            C(min_r1 : max_r2, :) = C(min_r1 : max_r2, :) - Y * WTC;
        end
    end
    Z = C;
end