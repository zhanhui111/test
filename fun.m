function fitness = fun(pop, hiddennum, inputnum, outputnum, p_train, t_train)
    % 检查输入参数维度
    numWeights = (inputnum * hiddennum) + hiddennum + (hiddennum * outputnum) + outputnum;
    if length(pop) ~= numWeights
        error('错误：预期的权重数量为 %d，但接收到 %d。请检查 pop 的列数。', numWeights, length(pop));
    end
    
    % 解码权值和阈值
    w1 = pop(1 : inputnum*hiddennum);
    B1 = pop(inputnum*hiddennum + 1 : inputnum*hiddennum + hiddennum);
    w2 = pop(inputnum*hiddennum + hiddennum + 1 : inputnum*hiddennum + hiddennum + hiddennum*outputnum);
    B2 = pop(end - outputnum + 1 : end);
    
    % 构建网络
    net = newff(p_train, t_train, hiddennum, {'tansig', 'purelin'}, 'trainlm');
    net.IW{1,1} = reshape(w1, hiddennum, inputnum);
    net.LW{2,1} = reshape(w2, outputnum, hiddennum);
    net.b{1} = reshape(B1, hiddennum, 1);
    net.b{2} = reshape(B2, outputnum, 1);  % 修正输出层偏置维度
    
    net.trainParam.showWindow = 0;
    net.trainParam.epochs = 0;
    
    % 预测（确保输入输出维度对齐）
    t_sim = sim(net, p_train);
    if size(t_sim, 1) ~= size(t_train, 1)
        error('网络输出维度与目标数据不匹配。');
    end
    
    % 计算均方根误差
    fitness = sqrt(mean((t_sim - t_train).^2));
end
