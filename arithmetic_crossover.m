function crossover_pop = arithmetic_crossover(selected, pc, sizepop)
    % 算术交叉，强制生成 sizepop 行
    crossover_pop = selected;  % 初始化
    for i = 1:2:sizepop-1
        if rand() < pc
            alpha = rand();
            parent1 = selected(i, :);
            parent2 = selected(i+1, :);
            crossover_pop(i, :) = alpha * parent1 + (1 - alpha) * parent2;
            crossover_pop(i+1, :) = alpha * parent2 + (1 - alpha) * parent1;
        end
    end
end