function [W_1, b_1, W_2, b_2,f,iter,time] = steepest_descent(W_1, b_1, W_2, b_2, X, Y, assistant_array, kmax,eps)
tic;
f = zeros(kmax,1);
alpha = 0.4;
gb2 = gb_2(X, Y, W_1, W_2, b_1, b_2);
gW2 = gW_2(X, Y, W_1, W_2, b_1, b_2);
gb1 = gb_1(X, assistant_array, W_1, W_2, b_1, b_2);
gW1 = gW_1(X, assistant_array, W_1, W_2, b_1, b_2);
for iter = 1:kmax
    f(iter) = fun3(X, assistant_array, W_1, W_2, b_1, b_2);
    f(iter)
    
    if norm(gb2) > eps
        b_2 = b_2 - alpha * gb2;
        gb2 = gb_2(X, Y, W_1, W_2, b_1, b_2);
    end
    if norm(gW2) > eps
        W_2 = W_2 - alpha * gW2;
        gW2 = gW_2(X, Y, W_1, W_2, b_1, b_2);
    end
    if norm(gb1) > eps
        b_1 = b_1 - alpha * gb1;
        gb1 = gb_1(X, assistant_array, W_1, W_2, b_1, b_2);
    end
    if norm(gW1) > eps
        W_1 = W_1 - alpha * gW1;
        gW1 = gW_1(X, assistant_array, W_1, W_2, b_1, b_2);
    end

    disp(iter);
end
time = toc;
f = f(1:iter);
end