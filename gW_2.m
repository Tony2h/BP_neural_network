%Ŀ�꺯����W_2��ƫ����
function gW_2 = gW_2(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp2 = ReLU(W_1 * X' + b_1);%��ʱ������ΪReLU���������sizeΪ l*n
tmp1 = W_2 * tmp2 + b_2;%��ʱ������Ϊsoftmax�������룬sizeΪ k*n
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