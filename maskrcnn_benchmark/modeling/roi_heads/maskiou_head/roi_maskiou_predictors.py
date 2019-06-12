from torch import nn
from maskrcnn_benchmark.modeling import registry


@registry.ROI_MASKIOU_PREDICTOR.register("MaskIoUPredictor")
class MaskIoUPredictor(nn.Module):
    def __init__(self, cfg):
        super(MaskIoUPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.maskiou = nn.Linear(1024, num_classes)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou


def make_roi_maskiou_predictor(cfg):
    func = registry.ROI_MASKIOU_PREDICTOR['MaskIoUPredictor']
    return func(cfg)
