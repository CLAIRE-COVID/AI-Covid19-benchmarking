from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
from utils.models import add_net_to_params
from utils.saver import Saver

def norm_01(x):
    x = x.detach().cpu()
    for b in range(x.shape[0]):
        x[b] = (x[b] - x[b].min())/(x[b].max() - x[b].min())
    return x

class Trainer:

    def __init__(self, args):
        # Store args
        self.args = args

    def train(self, datasets):
        # Get args
        args = self.args
        saver = args.saver
        log_every = args.log_metrics_every
        plot_every = args.log_plots_every
        save_every = args.save_every
        # Compute splits names
        splits = list(datasets.keys())

        # Setup model
        args.return_features = True
        module = getattr(models, args.model)
        net = getattr(module, "Model")(vars(args))
        # Check resume
        if args.resume is not None:
            net.load_state_dict(Saver.load_state_dict(args.resume))
        # Read features size
        net.eval()
        with torch.no_grad():
            feat_size = net(datasets['train'][0][0].unsqueeze(0)).shape
        # Move to device
        net.to(args.device)
        # Add network to params
        add_net_to_params(net, args, 'net')
        
        # Prepare args for decoder
        dec_args = {}
        dec_args['data_size'] = feat_size[2]
        dec_args['data_channels'] = feat_size[1]
        dec_args['target_size'] = datasets['train'][0][0].shape[1]
        dec_args['target_channels'] = datasets['train'][0][0].shape[0]
        # Create decoder
        module = getattr(models, args.decoder)
        decoder = getattr(module, "Model")(dec_args)
        # Move to device
        decoder.to(args.device)

        # Setup data loader
        loaders = {
            s: DataLoader(datasets[s], batch_size=args.batch_size,
                shuffle=(s == 'train'),
                num_workers=(args.workers if not args.overfit_batch else 0),
                drop_last=True)
            for s in splits
        }

        # Optimizer params
        optim_params = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if args.optim == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif args.optim == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
        # Create optimizer
        optim_class = getattr(torch.optim, args.optim)
        optim = optim_class(params=list(net.parameters()) + list(decoder.parameters()), **optim_params)
        # Configure LR scheduler
        scheduler = None
        if args.reduce_lr_every is not None:
            print("Setting up LR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, args.reduce_lr_every, args.reduce_lr_factor
            )

        # Initialize output metrics
        result_metrics = {s: {} for s in splits}
        # Process each epoch
        try:
            for epoch in range(args.epochs):
                # Process each split
                for split in splits:
                    # Epoch metrics
                    epoch_metrics = {}
                    # Set network mode
                    if split == 'train':
                        net.train()
                        decoder.train()
                        torch.set_grad_enabled(True)
                    else:
                        net.eval()
                        decoder.eval()
                        torch.set_grad_enabled(False)
                    # Process each batch
                    dl = loaders[split]
                    pbar = tqdm(dl, leave=False)
                    for batch_idx, (inputs,_) in enumerate(pbar):
                        # Compute step
                        step = (epoch * len(dl)) + batch_idx
                        # Set progress bar description
                        pbar_desc = f'{split}, epoch {epoch+1}'
                        if split == 'train':
                            pbar_desc += f', step {step}'
                        pbar.set_description(pbar_desc)
                        # Move to device
                        inputs = inputs.to(args.device)
                        # Forward step
                        outputs = net(inputs)
                        outputs = decoder(outputs)
                        # Check NaN
                        if torch.isnan(outputs).any():
                            raise FloatingPointError('Found NaN values')
                        # Compute loss
                        loss = F.mse_loss(outputs, inputs)
                        # Optimize
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        # Initialize metrics
                        metrics = {'loss': loss.item()}
                        # Add metrics to epoch results
                        for k, v in metrics.items():
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                        # Log metrics
                        if step % log_every == 0:
                            for k, v in metrics.items():
                                saver.dump_metric(v, step, split, k, 'batch')
                        # Plot stuff
                        if step % plot_every == 0:
                            # Log inputs/outputs
                            saver.dump_batch_image(norm_01(inputs), step, split, 'inputs')
                            saver.dump_batch_image(norm_01(outputs), step, split, 'outputs')
                    # Epoch end: compute epoch metrics 
                    epoch_loss = sum(epoch_metrics['loss'])/len(epoch_metrics['loss'])
                    # Print to screen
                    pbar.close()
                    print(f'{split}, {epoch+1}: loss={epoch_loss:.4f}')
                    # Dump to saver
                    saver.dump_metric(epoch_loss, epoch, split, 'loss', 'epoch')
                    # Add to output results
                    result_metrics[split]['loss'] = result_metrics[split]['loss'] + [epoch_loss] if 'loss' in result_metrics[split] else [epoch_loss]
                # Check LR scheduler
                if scheduler is not None:
                    scheduler.step()
                # Save checkpoint
                if (epoch+1) % save_every == 0:
                    saver.save_model(net, args.model, epoch+1)
        except KeyboardInterrupt:
            pass
        except FloatingPointError as err:
            print(f'Error: {err}')
        # Return
        return net, result_metrics
