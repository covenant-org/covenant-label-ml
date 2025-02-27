import os
import logging
import mlflow

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import YOLOModel, YOLO_MODEL_ROOT
from control_models.polygon_labels import PolygonLabelsModel
from control_models.sam import SAMModel


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Register available model classes
available_model_classes = [
    PolygonLabelsModel,
]
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "").split(",")
if type(MLFLOW_MODEL_URI) is str:
    MLFLOW_MODEL_URI = [MLFLOW_MODEL_URI]


class NewModel(LabelStudioMLBase):
    """Covenant ML Backend model
    """

    def download_mlflow_models(self, models_uris):
        for uri in models_uris:
            if len(uri) == 0:
                continue
            mlflow.artifacts.download_artifacts(
                uri,
                dst_path=os.path.join(YOLO_MODEL_ROOT, uri.split("/")[-1]))

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "yolo")
        self.download_mlflow_models(MLFLOW_MODEL_URI)

    def detect_control_models(self) -> List[YOLOModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(
                        f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict_yolo(self, tasks: List[Dict]):
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:

            regions = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region["score"]
                          for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def predict_sam2(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        control = None
        for ctl in self.label_interface.controls:
            if ctl.tag == "PolygonLabels":
                control = ctl
                break
        if control is None:
            raise ValueError(
                f"PolygonLabels control tag not found in the label config:\n{self.label_config}"
            )
        model = SAMModel.create(self, os.path.join(
            YOLO_MODEL_ROOT, "sam2_hiera_tiny.pt"), "tiny", control)
        for task in tasks:
            path = model.get_path(task)
            result = model.predict(path, context)
            prediction = {
                "result": [result],
                "score": result["score"],
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        print(predictions)

        return ModelResponse(predictions=predictions)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}"
        )

        if not context or not context.get('result'):
            logger.info("Doing preannotation prediction")
            # if there is no context, no interaction has happened yet
            return self.predict_yolo(tasks)

        logger.info("Doing interactive prediction")
        return self.predict_sam2(tasks, context, **kwargs)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
