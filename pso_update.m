function [new_pop, new_V, new_pbest, new_gbest] = pso_update(...
    pop, V, c1, c2, w, Vmax, Vmin, popmin, popmax, fitness, pbest, gbest)
    [sizepop, dim] = size(pop);
    new_V = zeros(sizepop, dim);
    new_pop = zeros(sizepop, dim);
    new_pbest = pbest;
    new_gbest = gbest;
    
    % 更新个体最优和全局最优
    for i = 1:sizepop
        if fitness(i) < pbest(i).fitness
            new_pbest(i).position = pop(i, :);
            new_pbest(i).fitness = fitness(i);
        end
    end
    [min_fit, idx] = min(fitness);
    if min_fit < gbest.fitness
        new_gbest.position = pop(idx, :);
        new_gbest.fitness = min_fit;
    end
    
    % 更新速度和位置
    for i = 1:sizepop
        new_V(i, :) = w * V(i, :) ...
            + c1 * rand() * (new_pbest(i).position - pop(i, :)) ...
            + c2 * rand() * (new_gbest.position - pop(i, :));
        new_V(i, :) = max(min(new_V(i, :), Vmax), Vmin);
        new_pop(i, :) = pop(i, :) + new_V(i, :);
        new_pop(i, :) = max(min(new_pop(i, :), popmax), popmin);
    end
end
