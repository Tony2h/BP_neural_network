%ReLU函数，返回一个与输入参数同样大小的矩阵
function r = ReLU(x)
r = zeros(size(x));
r = max(x, r);
end