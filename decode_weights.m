function [w1, B1, w2, B2] = decode_weights(position, inputnum, hiddennum, outputnum)
    % 权重解码函数
    w1 = position(1 : inputnum * hiddennum);
    B1 = position(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    w2 = position(inputnum * hiddennum + hiddennum + 1 : ...
                 inputnum * hiddennum + hiddennum + hiddennum * outputnum);
    B2 = position(end - outputnum + 1 : end);
end
