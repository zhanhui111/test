function plot_results(T_train, T_sim1, T_test, T_sim2)
    % 可视化结果
    figure;
    subplot(2,1,1);
    plot(T_train, 'b-', 'LineWidth', 1.5); hold on;
    plot(T_sim1, 'r--', 'LineWidth', 1.5);
    legend('真实值', '预测值');
    title('训练集预测对比');
    
    subplot(2,1,2);
    plot(T_test, 'b-', 'LineWidth', 1.5); hold on;
    plot(T_sim2, 'r--', 'LineWidth', 1.5);
    legend('真实值', '预测值');
    title('测试集预测对比');
end
