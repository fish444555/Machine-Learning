function [y,grad]=dejong(x)
% this is an arbitrarily dimensioned function
% input: x - n x d input matrix (n d-dimensional vectors)
% global min is at x = [0,0,...,0]

y = sum(x.^2,2);
grad = 2.*x;



function [x_final] = mydescent (func_handle, flag_grad, x_init)

max_iter = 1500;
converge_threshold = 0.000001;
epsilon = 0.000001;
rate_init = 0.01;
rate_mult = 2;
rate_div = 2;

% 
% if flag_grad == 0%computed within mydescent.m
% 
%     pass
%     
% else
%     [m_y, m_grad] = feval(@func_handle, x_init);
% end

    
x = x_init;

for iter = 1: max_iter
    [m_y, m_grad] = feval(@func_handle, x);
    
    if exist('pre_y', 'var') == 1
        if (abs(m_y - pre_y)  / pre_y < converge_threshold) || (abs(m_grad - pre_grad) < epsilon)% converge condition
            break;
        end
    end
    
    x = x - rate_init * m_grad;
    pre_y = m_y;
end

x_final = x;