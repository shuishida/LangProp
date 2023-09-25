import torch


class WaypointL1Loss:
    def __init__(self, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss(reduction="none")
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def __call__(self, output, target):
        invaild_mask = target.ge(1000)
        output[invaild_mask] = 0
        target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = torch.tensor(0.)
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = torch.tensor(0.)
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = torch.tensor(0.)
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = torch.tensor(0.)
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3

