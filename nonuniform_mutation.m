function mutated_pop = nonuniform_mutation(pop, pm, gen, maxgen, popmin, popmax)
    % 均匀变异，保持行数不变
    mutated_pop = pop;
    for i = 1:size(pop,1)
        for j = 1:size(pop,2)
            if rand() < pm
                delta = (popmax - popmin) * (1 - gen/maxgen)^2;
                mutated_pop(i,j) = mutated_pop(i,j) + delta * (2*rand() - 1);
                mutated_pop(i,j) = min(max(mutated_pop(i,j), popmin), popmax);
            end
        end
    end
end