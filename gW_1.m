function gW_1 = gW_1(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
[L, M] = size(W_1);
[K, ~] = size(W_2);
tmp2 = ReLU(W_1 * X' + b_1);%临时变量，为ReLU函数输出，size为 l*n
tmp1 = W_2 * tmp2 + b_2;%临时变量，为softmax函数输入，size为 k*n
gW_1 = zeros(size(W_1));
for a = 1:L
    for b = 1:M
        for i = 1:n
            if tmp2(a, i)>0
                tmp3 = 0;
                for j = 1:K
                    tmp3 = tmp3 + exp(tmp1(j, i)) * W_2(j, a) * X(i, b);
                end
                gW_1(a, b) = gW_1(a, b) + W_2(assistant_array(i), a) * X(i, b)...
                    - tmp3 / sum(exp(tmp1(:, i)));%X(i, b)实际应为X'(b, i)
            end
        end
    end
end
gW_1 = -1/n * gW_1;
end