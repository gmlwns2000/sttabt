import os, cv2, torch, tqdm
import numpy as np
from PIL import Image
from torchvision import transforms

from trainer import dyvit_concrete_trainer as dyvit_concrete
from models import sparse_token as sparse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_test_images():
    #return 224x224 image center cropped
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    path = './samples/imagenet'
    imgs = []
    batch = []
    for file_name in sorted(os.listdir(path)):
        file_name = os.path.join(path, file_name)
        img = transform(Image.open(file_name))
        batch.append(transform_norm(img))
        imgs.append(img)
    batch = torch.stack(batch)
    return imgs, batch

def render_image(pil_img, masks, layer_filter=None):
    oimg = np.array(pil_img)
    H, W, C = oimg.shape
    imgs = [oimg]
    for ilayer in range(len(masks)):
        layer_input_mask = masks[ilayer]

        #render mask darken
        mask_img = cv2.resize(layer_input_mask.numpy().astype(np.uint8), dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
        img = oimg * mask_img.reshape(H, W, 1)
        img = img.astype(np.float64)
        img = oimg * 0.333 + img * 0.666
        img = img.astype(np.uint8)

        #render mask edge
        mask_img *= 255
        mask_img = cv2.Canny(mask_img, 100, 255)
        img = img * (1 - (mask_img.reshape(H, W, 1).astype(np.int32) // 255)).astype(np.uint8) + mask_img.reshape(H, W, 1)

        imgs.append(img)
    imgs = np.concatenate(imgs, axis=1)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    return imgs

def load_dyvit(checkpoint_path, base_rate=0.4):
    os.environ['PYTHONPATH'] = './thrid_party/DynamicViT/'
    from thrid_party.DynamicViT.models.dyvit import VisionTransformerDiffPruning
    SPARSE_RATIO = [0.4]
    PRUNING_LOC = [3, 6, 9]
    KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
    print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
    model = VisionTransformerDiffPruning(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, 
    )
    model.viz_mode = True

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])
    del state
    
    model.eval()

    return model, PRUNING_LOC

@torch.no_grad()
def vis_dyvit(checkpoint_path, imsize=224, patchsize=16, base_rate=0.4, interested_layer=None):
    print('vis_dyvit:', checkpoint_path)
    def get_keep_indices(decisions):
        keep_indices = []
        for i in range(3):
            if i == 0:
                keep_indices.append(decisions[i][0])
            else:
                keep_indices.append(keep_indices[-1].gather(1, decisions[i][0]))
        return keep_indices

    assert os.path.exists(checkpoint_path)

    imgs, batch = load_test_images()
    N, C, H, W = batch.shape
    T = imsize // patchsize
    model, pruning_loc = load_dyvit(checkpoint_path, base_rate=base_rate)

    output, decisions = model(batch)
    decisions = get_keep_indices(decisions)
    masks = [torch.ones((N, T, T)) for i in range(3)]
    for idec in range(3):
        indices = decisions[idec] #shape: (N, selected_tokens_len)
        mask = torch.zeros((N, T*T), dtype=torch.float32)
        mask = mask.scatter_(1, indices, 1)
        for i in range(3 if idec != 2 else 3):
            masks.append(mask.view(N, T, T))
    input_masks = torch.stack(masks, dim=1)

    plot_imgs = []
    for i, img in enumerate(imgs):
        plot_img = render_image(img, input_masks[i] if interested_layer is None else input_masks[i][interested_layer])
        path = f'./saves_plot/visualization_vit/{i}_dyvit.png'
        cv2.imwrite(path, plot_img)
        plot_imgs.append(plot_img)
        print('vis_dyvit:', path)
    
    return plot_imgs

def load_concrete(checkpoint_path, factor=4, p_logit=-1.5):
    model, teacher_model = dyvit_concrete.load_concrete_model(
        model_id='deit-small', factor=factor, p_logit=p_logit
    )
    model = model.eval() #type: sparse.ApproxSparseBertForSequenceClassification

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])
    del state
    
    model.bert.set_concrete_hard_threshold(0.5)
    return model

@torch.no_grad()
def vis_concrete(checkpoint_path, imsize=224, patchsize=16, factor=4, p_logit=-1.5, interested_layer=None):
    print('vis_concrete:', checkpoint_path)
    assert os.path.exists(checkpoint_path)
    
    imgs, batch = load_test_images()
    model = load_concrete(checkpoint_path, factor=factor, p_logit=p_logit)

    output = model(batch)
    input_masks = []
    for i, layer in enumerate(model.bert.encoder.layer):
        layer = layer #type: sparse.BertLayer
        T = imsize//patchsize
        layer_input_mask = layer.attention.get_attention().concrete_input_mask.view(-1, T*T+1)[:,1:].view(-1, T,T)
        input_masks.append(layer_input_mask)
        # if i == (len(model.bert.encoder.layer) - 1):
        #     print(layer.attention.output.dense.concrete_mask_hard.view(-1, T*T+1)[:,1:].view(-1, T,T)[0])
    input_masks = torch.stack(input_masks, dim=1)
    
    plot_imgs = []
    for i, img in enumerate(imgs):
        plot_img = render_image(img, input_masks[i] if interested_layer is None else input_masks[i][interested_layer])
        path = f'./saves_plot/visualization_vit/{i}_concrete.png'
        cv2.imwrite(path, plot_img)
        plot_imgs.append(plot_img)
        print('vis_concrete:', path)
    
    return plot_imgs

def label_stacked_plot(plot, labels_layer=None):
    left_pad = 224
    top_pad = 112
    left_stack_top_margin = 112 + top_pad
    left_stack_item_height = 224
    left_stack_item_right_pad = 24
    top_stack_left_margin = left_pad + 112
    top_stack_bottom_margin = 16
    top_stack_item_width = 224
    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2.0
    font_color = (0,0,0)
    font_thickness = 2
    grid_line_color = (255,255,255)
    grid_line_thickness = 3

    H, W, C = plot.shape
    img = np.ones((H + top_pad, W + left_pad, C), dtype=np.uint8)
    img *= 255
    img[top_pad:,left_pad:,:] = plot

    #render left_stack panel
    y = left_stack_top_margin
    i = 0
    while y < (H + top_pad):
        text = "STTABT" if (i % 2) == 0 else "DynamicViT"

        (label_width, label_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)
        img = cv2.putText(img, text, (left_pad - left_stack_item_right_pad - label_width, y + label_height // 2), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)

        y += left_stack_item_height
        i += 1
    
    #render top_stack panel
    x = top_stack_left_margin
    i = 0
    while x < (W + left_pad):
        text = "Original" if i == 0 else (f"Layer {i}" if (labels_layer is None or i >= len(labels_layer)) else labels_layer[i])

        (label_width, label_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)
        img = cv2.putText(img, text, (x - label_width // 2, top_pad - top_stack_bottom_margin), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)

        x += top_stack_item_width
        i += 1
    
    #render horizontal grid
    y = top_pad + left_stack_item_height
    while y < (H + top_pad):
        x_left = left_pad
        x_right = W + left_pad
        img = cv2.line(img, (x_left, y), (x_right, y), grid_line_color, grid_line_thickness, cv2.LINE_AA)
        y += left_stack_item_height
    
    #render vertical grid
    x = left_pad + top_stack_item_width
    while x < (W + left_pad):
        y_top = top_pad
        y_bot = H + top_pad
        img = cv2.line(img, (x, y_top), (x, y_bot), grid_line_color, grid_line_thickness, cv2.LINE_AA)
        x += top_stack_item_width
    
    return img

def combine_plots(plots_dyvit, plots_concrete, labels_layer=None):
    plot_stacks = []
    for i in range(len(plots_dyvit)):
        plot_stacks.append(plots_concrete[i])
        plot_stacks.append(plots_dyvit[i])
    plot_stacks = np.concatenate(plot_stacks, axis=0)
    
    return plot_stacks, label_stacked_plot(plot_stacks, labels_layer=labels_layer)

def main():
    def imwrite(name, img):
        path = f'./saves_plot/visualization_vit/{name}'
        cv2.imwrite(path+'.png', img)
        cv2.imwrite(path+'.jpg', img)
        print('main.imwrite:', path)
    def filter(lst, ids):
        return [item for i, item in enumerate(lst) if i in ids]
    
    #three layers
    interest_ids = [5, 7]
    interested_layers = [3,6,9]
    labels_layer = [str(i+1) for i in interested_layers]
    plots_dyvit = vis_dyvit(
        checkpoint_path='./thrid_party/DynamicViT/logs/dynamicvit_deit-s-0.4/checkpoint-29.pth',
        base_rate=0.4, interested_layer=interested_layers
    )
    plots_concrete = vis_concrete(
        checkpoint_path='./saves/dyvit-concrete-f4--1.5-nohard-e20-we14/checkpoint-19.pth',
        factor=4, p_logit=-1.5, interested_layer=interested_layers
    )
    plot_stacks, plot_stacks_labeled = combine_plots(filter(plots_dyvit, interest_ids), filter(plots_concrete, interest_ids), labels_layer)
    imwrite('vit_token_visualization_interested_layers_labeled', plot_stacks_labeled)
    interest_ids = [0, 4]
    plot_stacks, plot_stacks_labeled = combine_plots(filter(plots_dyvit, interest_ids), filter(plots_concrete, interest_ids), labels_layer)
    imwrite('vit_token_visualization_interested_layers_labeled_2', plot_stacks_labeled)

    #full layers
    interested_layers = None
    plots_dyvit = vis_dyvit(
        checkpoint_path='./thrid_party/DynamicViT/logs/dynamicvit_deit-s-0.4/checkpoint-29.pth',
        base_rate=0.4
    )
    plots_concrete = vis_concrete(
        checkpoint_path='./saves/dyvit-concrete-f4--1.5-nohard-e20-we14/checkpoint-19.pth',
        factor=4, p_logit=-1.5
    )
    
    plot_stacks, plot_stacks_labeled = combine_plots(plots_dyvit, plots_concrete)
    imwrite('vit_token_visualization', plot_stacks)
    imwrite('vit_token_visualization_labeled', plot_stacks_labeled)

    interest_ids = [5, 7]
    plot_stacks, plot_stacks_labeled = combine_plots(filter(plots_dyvit, interest_ids), filter(plots_concrete, interest_ids))
    imwrite('vit_token_visualization_interested', plot_stacks)
    imwrite('vit_token_visualization_interested_labeled', plot_stacks_labeled)

if __name__ == '__main__':
    main()