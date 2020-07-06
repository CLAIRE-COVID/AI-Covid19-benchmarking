import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import os

# Enable cudnn benchmark (can be disabled by suppress_random)
torch.backends.cudnn.benchmark = True

def to_01(ten: torch.FloatTensor):
    m = ten.min()
    M = ten.max()
    return (ten - m)/(M - m)

"""
The followings assume ten is in [0-1] range and CHANNEL FIRST
"""

def image_tensor_to_grid(ten: torch.FloatTensor, return_numpy=False,
                         return_image=False, keep_channels=False):
    """
    Convert a pytorch tensor with a batch of images into a grid. 8 els per row.
    """
    B, C, H, W = ten.shape
    grid = make_grid(ten, nrow=8, pad_value=1) #, normalize=True, scale_each=True)

    if keep_channels and C == 1:
        grid = grid[0:1]
    if return_numpy:
        grid = grid.to('cpu').detach().numpy()
    if return_image:
        grid = grid.to('cpu').detach().numpy()
        grid = (grid * 255).astype(np.uint8)
    return grid

def suppress_random(seed=1234):
    import random
    import numpy as np
    import torch
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Overlay traversability on image
def overlay_mask(image, mask, color=0, value=255):
    image = torch.tensor(np.array(image)).float()
    trav_t = TraversabilityMapTransform(image.shape[:-1])
    mask = trav_t(mask)
    mask = torch.tensor(np.array(mask)).float().unsqueeze(2).expand_as(image)
    mask = (255 - np.array(mask)) > 0
    for i in range(3):
        if i != color:
            mask[:,:,i] = 0
    image = np.array(image)
    image[mask] = value
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image

# Create output images
def create_outputs(saver, datasets, net, rgb_untransform, args):
    # Custom import
    import trainers.models
    # Read from args
    device = args.device
    # Create dummy trainer
    module = getattr(trainers.models, args.trainer)
    model_trainer = getattr(module, "Trainer")(net=net, args=vars(args))
    # Compute result dir
    result_dir = os.path.join(saver.path, 'results')
    result_dirs = {'train': os.path.join(result_dir, 'train'), 'test': os.path.join(result_dir, 'test')}
    os.makedirs(result_dirs['train'], exist_ok=True)
    os.makedirs(result_dirs['test'], exist_ok=True)
    # Set eval mode
    net.eval()
    # Process each dataset
    for split,dataset in datasets.items():
        # Initialize outputs
        outputs = []
        # Process each sample
        for i in range(len(dataset)):
            # Get input
            rgb,depth,scan,targets = dataset[i]
            # Get filename
            rgb_path = dataset.dataset.rgb_data[dataset.indices[i]]
            # Pre-process input
            rgb.unsqueeze_(0)
            rgb = rgb.to(device)
            depth.unsqueeze_(0)
            depth = depth.to(device)
            scan.unsqueeze_(0)
            scan = scan.to(device)
            targets.unsqueeze_(0)
            targets = targets.to(device)
            # Compute output (use split: test to prevent backpropagation)
            output,_ = model_trainer.forward_batch(rgb, depth, scan, targets, {'split': 'test'})
            # Compute max error
            output = output.cpu()
            targets = targets.cpu()
            error = (output - targets).abs().max().item()
            # Generate image
            rgb = rgb[0]
            rgb = rgb_untransform(rgb)
            rgb = (rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            rgb = Image.fromarray(rgb)
            # Create masked images
            targets = targets[0].tolist()
            target_mask = overlay_mask(rgb, targets, color=1, value=240)
            output_mask = overlay_mask(rgb, output.cpu()[0], color=0)
            rgb = np.array(rgb)
            target_mask = np.array(target_mask)
            output_mask = np.array(output_mask)
            collage = np.concatenate((rgb, target_mask, output_mask), axis=1)
            collage = Image.fromarray(collage)
            # Add to outputs
            outputs.append((rgb_path, collage, error, output.squeeze().tolist()))
        # Sort by error
        outputs = sorted(outputs, key=lambda x: x[2])
        # Write to file
        with open(os.path.join(result_dirs[split], 'outputs.csv'), 'w') as fp:
            for i,(rgb_path, collage, error, net_out) in enumerate(outputs):
                # Compute collage filename
                error = f'{error:.4f}'
                filename = f'{i:05d}_{error}.png'
                # Save collage
                out_path = os.path.join(result_dirs[split], filename)
                collage.save(out_path)
                # Add to csv
                net_out = [str(x) for x in net_out]
                fp.write(','.join([filename, rgb_path] + net_out) + '\n')