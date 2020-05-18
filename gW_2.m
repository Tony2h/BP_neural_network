%目标函数对W_2的偏导数
function gW_2 = gW_2(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp2 = ReLU(W_1 * X' + b_1);%临时变量，为ReLU函数输出，size为 l*n
tmp1 = W_2 * tmp2 + b_2;%临时变量，为softmax函数输入，size为 k*n
[L, M] = size(W_2);
gW_2 = zeros(size(W_2));
for a = 1:L
    for b = 1:M
        for i = 1:n
            gW_2(a, b) = gW_2(a, b) + tmp2(b, i) * ( Y(i, a) + exp(tmp1(a, i)) / sum(exp(tmp1(:, i))) );
        end
    end
end
gW_2 = -1/n .* gW_2;
end