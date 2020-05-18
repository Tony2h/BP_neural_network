function gb_1 = gb_1(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
[L, ~] = size(b_1);
[K, ~] = size(b_2);
tmp2 = ReLU(W_1 * X' + b_1);%��ʱ������ΪReLU���������sizeΪ l*n
tmp1 = W_2 * tmp2 + b_2;%��ʱ������Ϊsoftmax�������룬sizeΪ k*n
gb_1 = zeros(size(b_1));
for k = 1:L
    for i = 1:n
        if tmp2(k, i)>0
            tmp3 = 0;
            for j = 1:K
                tmp3 = tmp3 + exp(tmp1(j, i)) * W_2(j, k);%��ʱ�������ۼ�����log������ƫ���ķ���
            end
            gb_1(k) = gb_1(k) + W_2(assistant_array(i), k) - tmp3/sum(exp(tmp1(:, i)));
        end
    end
end
gb_1 = -1/n * gb_1;
end