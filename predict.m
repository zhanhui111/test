function T_sim = predict(net, p_data, ps_output)
    % 预测函数（含反归一化）
    t_sim = sim(net, p_data);
    T_sim = mapminmax('reverse', t_sim, ps_output); % 反归一化
end
