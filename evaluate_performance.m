function [R1, R2, mae1, mae2, mbe1, mbe2, error1, error2] = evaluate_performance(T_train, T_test, T_sim1, T_sim2)
    % 性能评估函数
    error1 = T_sim1 - T_train;
    error2 = T_sim2 - T_test;
    
    R1 = corr(T_train', T_sim1')^2;  % 训练集 R²
    R2 = corr(T_test', T_sim2')^2;   % 测试集 R²
    mae1 = mean(abs(error1));        % 训练集 MAE
    mae2 = mean(abs(error2));        % 测试集 MAE
    mbe1 = mean(error1);             % 训练集 MBE
    mbe2 = mean(error2);             % 测试集 MBE
end
