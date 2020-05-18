%��b_2��ƫ����
function gb_2 = gb_2(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp1 = W_2 * ReLU(W_1 * X' + b_1) + b_2;%��ʱ������Ϊsoftmax�������룬sizeΪl*n
gb_2 = sum(Y, 1)';
%sum()�����Ĳ���1��ʾ�����ÿ��Ԫ����͵õ�һ������������ǰֻ����˵�һ�����󵼣�gb_2��һ��kά������
for i = 1:n
    gb_2 = gb_2 - exp(tmp1(:, i)) ./ sum(exp(tmp1(:, i)));
end
gb_2 = -1/n * gb_2;
end