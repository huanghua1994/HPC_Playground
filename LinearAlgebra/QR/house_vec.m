function [v, b] = house_vec(x)
% Householder Vector, Van Loan book 4th edition Algorithm 5.1.1
    m = length(x);
    s = x(2 : m)' * x(2 : m);
    v = [1; x(2 : m)];
    if ((s == 0) && (x(1) >= 0))
        b = 0;
    elseif ((s == 0) && (x(1) < 0))
        b = -2;
    else
        mu = sqrt(x(1) * x(1) + s);
        if (x(1) <= 0)
            v(1) = x(1) - mu;
        else
            v(1) = -s / (x(1) + mu);
        end
        v1_2 = v(1) * v(1);
        b = 2 * v1_2 / (s + v1_2);
        v = v / v(1);
    end
end