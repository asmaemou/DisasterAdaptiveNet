import timeit
import torch
from torch.utils import data as torch_data

from utils import losses, measurers, metrics, models
from utils.datasets_idabd import IdaBDDataset


def model_evaluation(net, cfg, device, run_type, epoch_float, global_step):
    dataset = IdaBDDataset(cfg, run_type=run_type, disable_augmentations=True)

    dataloader_kwargs = {
        "batch_size": 1,
        "num_workers": 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    criterion = losses.ComboLoss(weights=cfg.TRAINER.LOSS.WEIGHTS)
    class_weights = losses.loss_class_weights(cfg.TRAINER.LOSS.CLASS_WEIGHTS, dataset.get_class_counts())

    m_total, m_loc, m_dmg = measurers.AverageMeter(), measurers.AverageMeter(), measurers.AverageMeter()
    m_dice = measurers.AverageMeter()

    start = timeit.default_timer()

    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, msk = batch["img"].to(device), batch["msk"].to(device)

            if cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
                x_cond = batch["cond_id"].to(device)
                logits = net(x, x_cond)
            else:
                logits = net(x)

            loss_loc = criterion(logits[:, 0], msk[:, 0].long()) * class_weights[0]

            loss_dmg = torch.tensor([0.0], device=device)
            for c in range(1, logits.size(1)):
                loss_dmg = loss_dmg + criterion(logits[:, c], msk[:, c].long()) * class_weights[c]

            loss = loss_loc + loss_dmg

            y_hat = torch.sigmoid(logits[:, 0])
            dice_sc = 1 - metrics.dice_round(y_hat, msk[:, 0])

            m_loc.update(loss_loc.item(), x.size(0))
            m_dmg.update(loss_dmg.item(), x.size(0))
            m_total.update(loss.item(), x.size(0))
            m_dice.update(dice_sc, x.size(0))

    elapsed = timeit.default_timer() - start
    print(
        f"[{run_type}] loss={m_total.avg:.6f}, "
        f"loc={m_loc.avg:.6f}, dmg={m_dmg.avg:.6f}, "
        f"dice={m_dice.avg:.6f}, time={elapsed:.2f}s"
    )

    return -m_total.avg