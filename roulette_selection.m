function selected = roulette_selection(pop, fitness, sizepop)
    % 轮盘赌选择，强制返回 sizepop 行
    probabilities = 1./(1 + fitness);  % 适应度越高概率越大
    probabilities = probabilities / sum(probabilities);
    selected_indices = randsample(size(pop,1), sizepop, true, probabilities);
    selected = pop(selected_indices, :);  % 确保返回 sizepop 行
end
