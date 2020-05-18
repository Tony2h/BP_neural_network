%softmax函数，返回一个与输入参数同样大小的矩阵
function s = softmax(x)
[K, N] = size(x);
s = zeros(K, N);
for i=1:K
    for j = 1:N
        s(i, j) = exp(x(i, j)) / sum(exp(x(:, j)));%逐项计算
    end
end
end