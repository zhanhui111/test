function [R1, R2, mae1, mae2, mbe1, mbe2, error1, error2] = main()
    % 清空环境
    clc;
    clear;
    close all;
    
    %% 数据准备
    res = xlsread('predict.xls');
    numRows = size(res, 1);
    train_size = min(2936, numRows - 1);
    
    P_train = res(1:train_size, 1:16)';
    T_train = res(1:train_size, 17)';
    P_test = res(train_size+1:end, 1:16)';
    T_test = res(train_size+1:end, 17)';
    
    % 数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test  = mapminmax('apply', P_test, ps_input);
    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test  = mapminmax('apply', T_test, ps_output);
    
    %% 神经网络参数
    inputnum  = size(p_train, 1);   % 输入层节点数
    hiddennum = 5;                  % 隐藏层节点数
    outputnum = size(t_train, 1);   % 输出层节点数
    
    %% 混合优化参数
    maxgen = 100;            
    sizepop = 20;            % 种群规模
    pc = 0.85;              
    pm = 0.15;              
    c1 = 1.5;               
    c2 = 1.7;               
    w_max = 0.9;            
    w_min = 0.4;            
    popmax = 5.0;           
    popmin = -5.0;          
    Vmax = 0.2 * (popmax - popmin); 
    Vmin = -Vmax;           
    
    %% 初始化种群、速度、个体最优和全局最优
    numWeights = (inputnum * hiddennum) + hiddennum + (hiddennum * outputnum) + outputnum;
    pop = rand(sizepop, numWeights) * (popmax - popmin) + popmin;
    V = zeros(sizepop, numWeights);
    fitness = inf(1, sizepop);
    
    % 初始化 pbest 和 gbest
    pbest = repmat(struct('position', [], 'fitness', []), sizepop, 1);
    for i = 1:sizepop
        pbest(i).position = pop(i, :);
        pbest(i).fitness = inf;
    end
    gbest.position = [];
    gbest.fitness = inf;

    %% GA-PSO协同优化
    best_fitness_history = zeros(1, maxgen);  % 初始化适应度历史记录数组
    
    for gen = 1:maxgen
        % ... 此处是计算适应度的代码 ...
         % 计算适应度
        temp_fitness = zeros(1, sizepop);
        parfor i = 1:sizepop
            temp_fitness(i) = fun(pop(i, :), hiddennum, inputnum, outputnum, p_train, t_train);
        end
        fitness = temp_fitness;
        
        % 更新个体最优
        for i = 1:sizepop
            if fitness(i) < pbest(i).fitness
                pbest(i).position = pop(i, :);
                pbest(i).fitness = fitness(i);
            end
        end
        
        % 更新全局最优
        [min_fit, idx] = min(fitness);
        if min_fit < gbest.fitness
            gbest.position = pop(idx, :);
            gbest.fitness = min_fit;
        end
        best_fitness_history(gen) = gbest.fitness;  % 记录当前代的最优适应度
        
        % ... 遗传和粒子群操作代码 ...
         % GA阶段（省略详细实现）
% 关键：确保 mutated_pop 的行数为 sizepop
        selected = roulette_selection(pop, fitness, sizepop);   % 修正选择逻辑
        crossover_pop = arithmetic_crossover(selected, pc, sizepop);  % 强制生成 sizepop 行
        mutated_pop = nonuniform_mutation(crossover_pop, pm, gen, maxgen, popmin, popmax);
        
        % PSO阶段（省略详细实现）
        w = w_max - (w_max - w_min) * (gen / maxgen);
        for i = 1:sizepop
            % 更新速度
            V(i, :) = w * V(i, :) + ...
                c1 * rand() * (pbest(i).position - pop(i, :)) + ...
                c2 * rand() * (gbest.position - pop(i, :));
            V(i, :) = min(max(V(i, :), Vmin), Vmax);  % 速度限制
            
            % 更新位置
            mutated_pop(i, :) = mutated_pop(i, :) + V(i, :);
            mutated_pop(i, :) = min(max(mutated_pop(i, :), popmin), popmax);  % 位置限制
        end
        
        % 步骤6: 更新种群（确保 mutated_pop 是 sizepop 行）
        if size(mutated_pop, 1) ~= sizepop
            error('变异后的种群行数错误：应为 %d，实际为 %d', sizepop, size(mutated_pop,1));
        end
        pop = mutated_pop;
    end

    %% 最优权值解码和网络训练（代码保持不变）
    %% 最优权值解码
    [w1, B1, w2, B2] = decode_weights(gbest.position, inputnum, hiddennum, outputnum);
    
    %% 神经网络训练与测试（修复：直接构建网络）
    net = newff(p_train, t_train, hiddennum, {'tansig', 'purelin'}, 'trainlm');
    net.IW{1,1} = reshape(w1, hiddennum, inputnum);
    net.LW{2,1} = reshape(w2, outputnum, hiddennum);
    net.b{1} = reshape(B1, hiddennum, 1);
    net.b{2} = reshape(B2, outputnum, 1);
    
    net = train(net, p_train, t_train);
    
    %% 预测与结果评估
    T_sim1 = predict(net, p_train, ps_output);
    T_sim2 = predict(net, p_test, ps_output);
    
    [R1, R2, mae1, mae2, mbe1, mbe2, error1, error2] = evaluate_performance(T_train, T_test, T_sim1, T_sim2);
    
    %% 可视化结果（新增适应度曲线）
    % 训练集和测试集预测对比图（原有代码）
    plot_results(T_train, T_sim1, T_test, T_sim2);
    
    % 训练集散点图（新增）
    figure;
    scatter(T_train, T_sim1, 'b', 'filled');
    hold on;
    plot([min(T_train), max(T_train)], [min(T_train), max(T_train)], 'r--', 'LineWidth', 1.5);
    title('训练集预测散点图');
    xlabel('真实值');
    ylabel('预测值');
    legend('预测值', '理想预测线', 'Location', 'northwest');
    grid on;
    hold off;

    % 测试集散点图（新增）
    figure;
    scatter(T_test, T_sim2, 'b', 'filled');
    hold on;
    plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r--', 'LineWidth', 1.5);
    title('测试集预测散点图');
    xlabel('真实值');
    ylabel('预测值');
    legend('预测值', '理想预测线', 'Location', 'northwest');
    grid on;
    hold off;

    % 适应度曲线（新增）
    figure;
    plot(1:maxgen, best_fitness_history, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    title('最佳适应度随迭代次数的变化');
    xlabel('进化代数');
    ylabel('最佳适应度值');
    grid on;
     % 计算误差
    error1 = sqrt(sum((T_sim1 - T_train).^2) / length(T_sim1));
    error2 = sqrt(sum((T_sim2 - T_test).^2) / length(T_sim2));
    disp(['训练集 RMSE: ', num2str(error1)]);
    disp(['测试集 RMSE: ', num2str(error2)]);
    % R2
    R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
    R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test))^2;

    disp(['训练集数据的R2为：', num2str(R1)])
    disp(['测试集数据的R2为：', num2str(R2)])

    % MAE
    M = length(T_train);  % 训练集数据点数量
    N = length(T_test);   % 测试集数据点数量
    mae1 = sum(abs(T_sim1 - T_train), 2)' ./ M ;
    mae2 = sum(abs(T_sim2 - T_test ), 2)' ./ N ;

    disp(['训练集数据的MAE为：', num2str(mae1)])
    disp(['测试集数据的MAE为：', num2str(mae2)])

    % MBE
    mbe1 = sum(T_sim1 - T_train, 2)' ./ M ;
    mbe2 = sum(T_sim2 - T_test , 2)' ./ N ;

    disp(['训练集数据的MBE为：', num2str(mbe1)])
    disp(['测试集数据的MBE为：', num2str(mbe2)])
    
end
