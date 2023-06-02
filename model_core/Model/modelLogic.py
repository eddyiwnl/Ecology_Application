from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Used for standard transformations in the ImageDataset
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1


class Model(pl.LightningModule):
    def __init__(self, num_classes=6, score_threshold=.375, nms_iou_threshold=.3):
        super().__init__()

        self.reverse_label_dict = {1: 'Unknown', 2: 'Amphipoda',
                                   3: 'Polychaeta', 4: 'Ostracoda', 5: 'Unknown', 0: 'Background'}
        self.label_dict = {v: k for k, v in self.reverse_label_dict.items()}
        self.model = self.get_object_detection_model(num_classes)
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def get_object_detection_model(self, num_classes):
        """
        Inputs
            num_classes: int
                Number of classes to predict. Must include the
                background which is class 0 by definition!
        """
        pretrained_model = fasterrcnn_resnet50_fpn_v2()

        in_feats = pretrained_model.roi_heads.box_predictor.cls_score.in_features
        pretrained_model.roi_heads.box_predictor = FastRCNNPredictor(
            in_feats, num_classes)
        return pretrained_model

    def forward(self, torch_data_loader):
        """
        Get model predictions from a torch_data_loader
        Inputs
            torch_data_loader: PyTorch DataLoader Object
        Returns:
            List(Dict("boxes", "labels", "scores"))
        """
        image_names, predictions = self.predict(torch_data_loader)
        return image_names, self.decode_predictions(predictions)

    def predict(self, data_loader):
        """
        Gets the predictions for a batch of data.
        Inputs
            model: torch model
            data_loader: torch Dataloader
        Returns
            images_names: list
                List of the image names that were predicted on
            predictions: list
                List of dicts containing the predictions for the 
                bounding boxes, labels and confidence scores.
        """
        image_names = []
        predictions = []
        for batch in data_loader:
            _, X_name, P = self.predict_batch(batch)
            for i in range(len(X_name)):
                image_names.append(X_name[i])
                predictions.append(P[i])
        return image_names, predictions

    @torch.no_grad()
    def predict_batch(self, batch):
        """
        Gets the predictions for a batch of data.
        Inputs
            batch: tuple
                Tuple containing a batch from the Dataloader.
            model: torch model
            device: str
                Indicates which device (CPU/GPU) to use.
        Returns
            images: list
                List of tensors of the images.
            predictions: list
                List of dicts containing the predictions for the 
                bounding boxes, labels and confidence scores.
        """
        def unbatch(batch):
            X, X_name = batch
            X = [x.cpu() for x in X]
            return X, X_name

        X, X_name = unbatch(batch)
        predictions = self.model(X)
        return X, X_name, predictions

    def decode_predictions(self, image_predictions, keep_unknown=True):
        """
        Filter the predicted boxes by the score_threshold and nms_iou_threshold for every image
        """
        ret = []
        for image_prediction in image_predictions:
            pred_boxes = image_prediction["boxes"]
            pred_scores = image_prediction["scores"]
            pred_labels = np.array(list(map(
                lambda x: self.reverse_label_dict[x], image_prediction["labels"].cpu().numpy())))

            # Remove any low-score predictions.
            if self.score_threshold is not None:
                want = pred_scores > self.score_threshold
                pred_boxes = pred_boxes[want]
                pred_scores = pred_scores[want]
                pred_labels = pred_labels[want]

            # Remove any overlapping bounding boxes using NMS.
            if self.nms_iou_threshold is not None:
                want = torchvision.ops.nms(boxes=pred_boxes, scores=pred_scores,
                                           iou_threshold=self.nms_iou_threshold)
                pred_boxes = pred_boxes[want]
                pred_scores = pred_scores[want]
                pred_labels = pred_labels[want]

            ret.append({"boxes": pred_boxes.cpu().numpy().astype(int),
                        "labels": pred_labels,
                        "scores": pred_scores.cpu().numpy()})

        # Filter 'Unknown' class predictions
        if not keep_unknown and 'Unknown' in self.label_dict.keys():
            print('Removing an Unknown label')
            filtered = []
            for cur in ret:
                want = [i for i, x in enumerate(
                    cur['labels']) if x != 'Unknown']
                boxes = cur['boxes'][want]
                labels = cur['labels'][want]
                scores = cur['scores'][want]
                filtered.append(
                    {"boxes": boxes, "scores": scores, "labels": labels})
            ret = filtered
        return ret
