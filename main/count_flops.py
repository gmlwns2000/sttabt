import math

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

TOPK_FLOPS = 10
UNIQUE_FLOPS = 10

def bert(H, L=12, INTER_FACTOR=4, TOKENS=512, OCCUPY=[1.0 for _ in range(13)], CLS=4, HEAD=12, layer_mask=False):
    flop = 0
    #bert.encoder
    #bert.encoder.layer
    for l in range(L):
        T = TOKENS * OCCUPY[l]
        T_NEXT = TOKENS * OCCUPY[l+1]
        #layer.attention
        #layer.attention.self
        #layer.attention.self.query
        flop += T_NEXT*H*H*2    #matmul
        flop += T_NEXT*H        #bias
        #layer.attention.self.key
        flop += T*H*H*2
        flop += T*H
        #layer.attention.self.value
        flop += T*H*H*2
        flop += T*H
        #layer.attention.self.forward
        flop += T_NEXT*H*T*2                #attention matmul
        flop += T_NEXT*T                    #attention mask
        flop += T_NEXT*T*SOFTMAX_FLOPS      #softmax
        flop += T_NEXT*T*H*2                #context matmul
        #layer.attention.output
        flop += T_NEXT*H*H*2
        flop += T_NEXT*H
        flop += T_NEXT*H*LAYER_NORM_FLOPS
        #layer.intermediate
        flop += T_NEXT*H*H*INTER_FACTOR*2   #weight
        flop += T_NEXT*H*INTER_FACTOR       #bias
        flop += T_NEXT*H*ACTIVATION_FLOPS   #act
        #layer.output
        flop += T_NEXT*H*INTER_FACTOR*H*2
        flop += T_NEXT*H
        flop += T_NEXT*H*LAYER_NORM_FLOPS
        #update_layer_mask
        if layer_mask:
            flop += T*TOKENS*(HEAD+1) #mean
            flop += T*TOKENS*3
            flop += T*TOKENS + TOKENS
            flop += T*TOKENS
            flop += T*(TOKENS*math.log(TOKENS)*TOPK_FLOPS)
            flop += T*(TOKENS*math.log(TOKENS)*UNIQUE_FLOPS)
    #bert.pooler
    flop += H*CLS*2
    flop += CLS
    return flop

def sparse_bert(H, FACTOR, OCCUPY, TOKENS=512, LAYERS=12):
    assert len(OCCUPY) == LAYERS+1
    return bert(H//FACTOR, TOKENS=TOKENS, L=LAYERS) + bert(H, TOKENS=TOKENS, OCCUPY=OCCUPY, layer_mask=True, L=LAYERS)

if __name__ == '__main__':
    H = 768
    T = 512

    base = bert(H, TOKENS=T)
    print('approx_net, factor=8', bert(H//8, TOKENS=T) / base, 1/64)
    print('approx_net, factor=4', bert(H//4, TOKENS=T) / base, 1/16)
    occupy = [0.25 for _ in range(13)]
    occupy[0] = 1.0
    occupy[-1] = 1/T
    print('sparse_net, factor=4, occupy=0.25', sparse_bert(H, 8, occupy, TOKENS=T) / base)