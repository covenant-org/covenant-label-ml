import os
from uuid import uuid4
from pydantic import BaseModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from label_studio_ml.model import LabelStudioMLBase
from sam2.utils.misc import variant_to_config_mapping
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image
from typing import Optional, Dict
import numpy as np
import cv2
_sam_cache = {}


class SAMModel(BaseModel):
    model: SAM2ImagePredictor
    value: str
    to_name: str
    from_name: str
    control: ControlTag
    backend: LabelStudioMLBase
    label_map: Optional[Dict[str, str]] = {}

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def load_sam_model(file_path, mapping):
        if file_path not in _sam_cache:
            builder = build_sam2(
                variant_to_config_mapping[mapping],
                file_path
            )
            _sam_cache[file_path] = SAM2ImagePredictor(builder)

        return _sam_cache[file_path]

    @classmethod
    def create(cls, mlbackend: LabelStudioMLBase, path: str, mapping: str, control: ControlTag):
        from_name = control.name
        to_name = control.to_name[0]
        value = control.objects[0].value_name
        label_map = mlbackend.build_label_map(from_name, ["default"])

        return cls(
            model=cls.load_sam_model(path, mapping),
            value=value,
            to_name=to_name,
            from_name=from_name,
            control=control,
            backend=mlbackend,
            label_map=label_map
        )

    def get_path(self, task):
        task_path = task["data"].get(self.value)
        if task_path is None:
            raise ValueError(
                f"Can't load path using key '{self.value}' from task {task}"
            )
        if not isinstance(task_path, str):
            raise ValueError(f"Path should be a string, but got {task_path}")

        path = (
            task_path
            if os.path.exists(task_path)
            else get_local_path(task_path, task_id=task.get("id"))
        )
        return path

    def set_image(self, path):
        image = Image.open(path)
        image = np.array(image.convert("RGB"))
        self.model.set_image(image)

    def get_points_and_box(self, context):
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        point_coords = []
        point_labels = []
        input_box = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            if ctx_type == 'keypoint':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectangle':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(
                    box_width + x), int(box_height + y)]

        return (point_coords, point_labels, input_box)

    def create_polygons(self, masks, scores, selected_label, size):
        w, h = size
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        coords = []
        for obj in contours:
            for point in obj:
                coords.append([point[0][0], point[0][1]])
        coords = np.array(coords, dtype=np.int32)
        coords = cv2.approxPolyDP(coords, 1, True)
        curve = []
        for obj in coords:
            for point in obj:
                print(point)
                curve.append([point[0]/w, point[1]/h])
        curve = np.array(curve, dtype=np.float32) * 100
        label_id = str(uuid4())[:4]

        region = {
            "id": label_id,
            "from_name": self.from_name,
            "to_name": self.to_name,
            "type": "polygonlabels",
            "value": {
                "polygonlabels": [selected_label],
                "points": curve.tolist(),  # Converting the tensor to a list for JSON serialization
                "closed": True,
            },
            "original_width": w,
            "original_height": h,
            "score": prob,
        }
        return region

    def predict(self, path, context):
        self.set_image(path)
        point_coords, point_labels, input_box = self.get_points_and_box(
            context)
        point_coords = np.array(
            point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(
            point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(
            input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = self.model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        value = "default"
        for value in self.label_map:
            value = self.label_map[value]

        return self.create_polygons(masks, scores, value, (image_width, image_height))

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'
