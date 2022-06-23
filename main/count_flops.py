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

def bert(H, L=12, INTER_FACTOR=4, TOKENS=512, CLS=4, HEAD=4, layer_mask=False):
    flop = 0
    #bert.encoder
    #bert.encoder.layer
    for _ in range(L):
        #layer.attention
        #layer.attention.self
        #layer.attention.self.query
        flop += TOKENS*H*H*2
        #layer.attention.self.key
        flop += TOKENS*H*H*2
        #layer.attention.self.value
        flop += TOKENS*H*H*2
        #layer.attention.self.forward
        flop += TOKENS*H*TOKENS*2 #attention matmul
        flop += TOKENS*TOKENS #attention mask
        flop += TOKENS*TOKENS*SOFTMAX_FLOPS #softmax
        flop += TOKENS*TOKENS*H*2 #context matmul
        #layer.attention.output
        flop += TOKENS*H*H*2
        flop += TOKENS*H*LAYER_NORM_FLOPS
        #layer.intermediate
        flop += TOKENS*H*H*INTER_FACTOR*2
        flop += TOKENS*H*ACTIVATION_FLOPS
        #layer.output
        flop += TOKENS*H*INTER_FACTOR*H*2
        flop += TOKENS*H*LAYER_NORM_FLOPS
        #update layer mask
        if layer_mask:
            flop += TOKENS*TOKENS*(HEAD+1) #mean
            flop += TOKENS*TOKENS*3
            flop += TOKENS*TOKENS + TOKENS
            flop += TOKENS*TOKENS
            flop += TOKENS*(TOKENS*math.log(TOKENS)*TOPK_FLOPS)
            flop += TOKENS*(TOKENS*math.log(TOKENS)*UNIQUE_FLOPS)
    #bert.pooler
    flop += H*CLS*2
    return flop

H = 768
T = 512

base = bert(H, TOKENS=T)
print('approx_net, factor=8', bert(H//8, TOKENS=T) / base, 1/64)
print('approx_net, factor=4', bert(H//4, TOKENS=T) / base, 1/16)
print(
    'sparse_net, factor=4, occupy=0.25', 
    (bert(H//4, TOKENS=T) + bert(768, TOKENS=T*0.25, layer_mask=True)) / base
)