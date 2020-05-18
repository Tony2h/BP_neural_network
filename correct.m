%计算正确率
function correct_rate = correct(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
y = softmax(W_2 * ReLU(W_1 * X' + b_1) + b_2);
count = 0;
for i = 1:n
    if y(assistant_array(i), i) == max(y(:, i))
        count = count+1;
    end
end
correct_rate = count/n * 100;
end