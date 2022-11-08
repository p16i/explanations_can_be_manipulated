import os

import argparse

import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir


import pandas as pd

from tqdm import tqdm

def get_label_ix_from_nsid(nsid):
    df = pd.read_csv("./imagenet-label-mapping.csv")

    return df[df["imagenet-id"] == nsid].index.values[0]


def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def viz_heatmap(heatmap, xlabel_prefix=""):


    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    b = np.abs(heatmap).max()

    plt.imshow(heatmap, vmin=-b, vmax=b, cmap=my_cmap)
    plt.xticks([]); plt.yticks([])

    plt.xlabel(f"$\sum R_i = {heatmap.sum():.4f}$")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_iter', type=int, default=1500, help='number of iterations')
    argparser.add_argument('--seed_file', type=str, default='something.csv', help='seed file')
    argparser.add_argument('--lr', type=float, default=0.0002, help='lr')
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--data_dir', type=str, default='./data/imagenet', help='data-dir')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input'],
                           default='lrp')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # load model
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    vgg_model = torchvision.models.vgg16(pretrained=True)
    model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
    if method == ExplainingMethod.pattern_attribution:
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)

    slugs = args.seed_file.split("/")

    nsid = slugs[-1].replace(".csv", "")

    output_dir = "/".join(slugs[:-1] + ["_output", nsid, args.method])

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.seed_file)
    
    tbar = tqdm(total=args.num_iter*df.shape[0])

    arr_all_x_adv = []
    arr_all_expl = []

    label = get_label_ix_from_nsid(nsid)

    for rix, row in enumerate(df.to_dict("records")):
        orig_img = row["original"]
        target_img = row["target"]

        # load images
        x = load_image(data_mean, data_std, device, f"{args.data_dir}/{orig_img}")
        x_target = load_image(data_mean, data_std, device, f"{args.data_dir}/{target_img}")

        x_adv = x.clone().detach().requires_grad_()

        # produce expls
        org_expl, org_acc, ori_class_ix = get_expl(model, x, method, desired_index=label)
        org_expl = org_expl.detach().cpu()


        target_expl, target_acc, target_class_ix = get_expl(model, x_target, method)
        target_expl = target_expl.detach()
        
        optimizer = torch.optim.Adam([x_adv], lr=args.lr)


        arr_sample_x_adv = [
            x_adv.detach().cpu()
        ]

        total_relevance_scaling = org_expl.sum() / target_expl.sum()
        target_expl_rescaled = target_expl * total_relevance_scaling

        for i in range(args.num_iter):
            if args.beta_growth:
                model.change_beta(get_beta(i, args.num_iter))

            optimizer.zero_grad()

            # calculate loss
            adv_expl, adv_acc, _ = get_expl(model, x_adv, method, desired_index=label)
            loss_expl = F.mse_loss(
                adv_expl,
                # we make sure that the total relevance score is preserved.
                target_expl_rescaled
            )
            loss_output = F.mse_loss(adv_acc, org_acc.detach())
            total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output

            # update adversarial example
            total_loss.backward()
            optimizer.step()


            # clamp adversarial example
            # Note: x_adv.data returns tensor which shares data with x_adv but requires
            #       no gradient. Since we do not want to differentiate the clamping,
            #       this is what we need
            x_adv.data = clamp(x_adv.data, data_mean, data_std)

            if (i + 1) % 300 == 0 or i == args.num_iter - 1:
                arr_sample_x_adv.append(
                    x_adv.detach().cpu()
                )

            tbar.set_description("Sample {:2d} Iter {:4d}: Total Loss: {:.2e}, Expl Loss: {:.2e}, Output Loss: {:.2e}".format(rix, i, total_loss.item(), loss_expl.item(), loss_output.item()))

            tbar.update()

        # test with original model (with relu activations)
        model.change_beta(None)
        adv_expl, adv_acc, adv_class_ix = get_expl(model, x_adv, method, desired_index=label)

        # save results
        plot_overview(
            [x_target, x, x_adv],
            [target_expl, org_expl, adv_expl],
            data_mean, data_std,
            filename=f"{output_dir}/sample-{rix:03d}.png",
            captions=[
                f'Target Image (True: {int(target_class_ix)}, Pred: {torch.argmax(target_acc).detach().cpu()})',
                f'Original Image (True: {label}, Pred: {int(ori_class_ix.detach().cpu())})',
                f'Manipulated Image (True: {label}, Pred: {int(adv_class_ix.detach().cpu())})',
                f'Target Explanation (w.r.t Class {int(target_class_ix)})',
                f'Original Explanation (w.r.t Class {label})',
                f'Manipulated Explanation (w.r.t Class {label})'
            ]
        )

        arr_sample_x_adv = torch.cat(arr_sample_x_adv).detach().cpu()

        arr_sample_expl = torch.cat([
            org_expl,
            target_expl.detach().cpu(),
            adv_expl.detach().cpu(),
        ])

        arr_all_x_adv.append(arr_sample_x_adv)
        arr_all_expl.append(arr_sample_expl)

        break

    arr_all_x_adv = torch.stack(arr_all_x_adv)
    arr_all_expl = torch.stack(arr_all_expl)
    assert len(arr_all_x_adv.shape) == 5 and arr_all_x_adv.shape[0] == df.shape[0] and arr_all_x_adv.shape[2] == 3
    assert len(arr_all_expl.shape) == 4 and arr_all_expl.shape[0] == df.shape[0] and arr_all_expl.shape[1] == 3


    torch.save(arr_all_x_adv, f"{output_dir}/x-adv.pth")
    torch.save(arr_all_expl, f"{output_dir}/explanations.pth")
    print(f"Check results at {output_dir}")


if __name__ == "__main__":
    main()