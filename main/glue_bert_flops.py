from trainer import glue_base as glue
from utils import sparse_flops_calculation as calc
from main.plot.constants import GLUE_SUBSETS
import tqdm,json

def main():
    data = {}

    for subset in GLUE_SUBSETS:
        trainer = glue.GlueAttentionApproxTrainer(
            subset, factor=4, device='cpu', world_size=1, wiki_train=False,
            enable_plot=False
        )
        testset = trainer.test_dataloader
        flops_sum = 0
        flops_count = 0
        for batch in tqdm.tqdm(testset):
            N, T = batch['input_ids'].shape
            base_config = calc.ModelConfig(
                num_layer=12,
                num_heads=12,
                hidden_size=768,
                intermediate_size=768*4,
                seq_len=T,
                arch='bert',
                token_occupies=None
            )
            flops = calc.flops_sparse_bert_model(base_config) * 1e-9
            flops_sum += flops * N
            flops_count += N
        flops_sum /= flops_count
        data[subset] = flops_sum
        print(f"main: {subset} {flops_sum} GFLOPs")
    
    print('main:', data)

    with open('saves_plot/glue_bert_flops.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()