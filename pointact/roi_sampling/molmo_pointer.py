"""MolmoPoint-8B wrapper: an instruction plus camera frames in, pixel detections out.

Runs in ``envs/molmo`` (transformers 4.57.1), never in the training env. The model is
frozen and detached from PointAct: it only decides *where* the point budget is spent, and
its output never reaches the policy as an input.

MolmoPoint does not emit text coordinates. It generates ``<PATCH>``/``<SUBPATCH>``/
``<LOCATION>`` grounding tokens that select a visual token and then refine within it, so
decoding needs preprocessor metadata (``return_pointing_metadata=True``) and generation
needs the model's own logits processor to stay on the valid-token manifold. Both are
handled here.

One request can carry **several images**, and each returned point says which image it
landed in. That is why a frame costs one forward rather than one per camera: the left and
right agentviews go in together, and the caller lifts whichever views produced a point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Field order of the tuples ``extract_image_points`` returns. The model card's runnable
#: example documents ``[object_id, image_num, x, y]``, while its prose says
#: ``(image_id, object_id, x, y)`` -- the two disagree, and picking wrong silently swaps
#: which camera a detection is attributed to. :meth:`MolmoPointer.verify_point_order`
#: settles it against the actual checkpoint; call it once before trusting a fresh cache.
POINT_FIELD_ORDER = ("object_id", "image_num", "x", "y")


@dataclass
class Detection:
    """One point, in the pixel coordinates of the image it was found in."""

    image_num: int
    object_id: int
    x: float
    y: float


class MolmoPointer:
    """Lazily-loaded MolmoPoint-8B, batched over (frame, query) requests."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        point_field_order: tuple[str, ...] = POINT_FIELD_ORDER,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.point_field_order = tuple(point_field_order)

        # local_files_only: compute nodes have no internet, and the failure mode without it
        # is a hang rather than an error (docs/clusters/jean-zay.md).
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=True, dtype="auto", device_map=device,
            local_files_only=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True, padding_side="left", local_files_only=True,
        )

    def _messages(self, images: list[np.ndarray], prompt: str) -> list[dict]:
        return [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
                       + [{"type": "image", "image": im} for im in images],
        }]

    def point(
        self, image_sets: list[list[np.ndarray]], prompts: list[str]
    ) -> list[list[Detection]]:
        """Point at ``prompts[i]`` within ``image_sets[i]``.

        Args:
            image_sets: one list of RGB uint8 HxWx3 arrays per request; every image in a
                set is searched and the returned detections say which one they came from.
            prompts: one natural-language pointing instruction per request.

        Returns:
            One list of :class:`Detection` per request, possibly empty when the model
            declines to point (it has an explicit no-more-points class, so an empty result
            is a real answer, not a parse failure).
        """
        if len(image_sets) != len(prompts):
            raise ValueError(f"{len(image_sets)} image sets vs {len(prompts)} prompts")
        if not image_sets:
            return []

        torch = self.torch
        conversations = [self._messages(ims, p) for ims, p in zip(image_sets, prompts)]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        prompt_len = inputs["input_ids"].size(1)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs,
                logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=self.max_new_tokens,
            )
        texts = self.processor.post_process_image_text_to_text(
            out[:, prompt_len:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        results = []
        for i, text in enumerate(texts):
            raw = self.model.extract_image_points(
                text,
                _per_sample(metadata["token_pooling"], i),
                _per_sample(metadata["subpatch_mapping"], i),
                _per_sample(metadata["image_sizes"], i),
            )
            results.append(self._to_detections(raw, n_images=len(image_sets[i])))
        return results

    def _to_detections(self, raw, n_images: int) -> list[Detection]:
        fields = self.point_field_order
        out = []
        for row in np.asarray(raw, dtype=np.float64).reshape(-1, 4):
            rec = dict(zip(fields, row))
            image_num = int(rec["image_num"])
            if not 0 <= image_num < n_images:
                # Almost certainly POINT_FIELD_ORDER is wrong for this checkpoint. Drop the
                # point rather than attributing it to a camera it did not come from -- a
                # mislabelled view lifts to a plausible-looking but wrong 3D anchor.
                logger.warning(
                    "point image_num=%d outside [0, %d) -- check POINT_FIELD_ORDER "
                    "(see verify_point_order)", image_num, n_images)
                continue
            out.append(Detection(image_num=image_num, object_id=int(rec["object_id"]),
                                 x=float(rec["x"]), y=float(rec["y"])))
        return out

    def verify_point_order(self, size: int = 224) -> tuple[str, ...]:
        """Settle the ``[object_id, image_num, ...]`` ambiguity against this checkpoint.

        Sends two synthetic images where the queried object exists in the SECOND one only,
        so a correct reading must report ``image_num == 1``. Returns the field order that
        holds, and logs a warning if it is not the configured one.
        """
        blank = np.full((size, size, 3), 245, dtype=np.uint8)
        withdot = blank.copy()
        c, r = size // 2, size // 8
        withdot[c - r:c + r, c - r:c + r] = (200, 30, 30)  # one unmistakable red square

        raw = None
        try:
            inputs_raw = self._raw_points([blank, withdot], "Point to the red square")
            raw = np.asarray(inputs_raw, dtype=np.float64).reshape(-1, 4)
        except Exception as exc:  # noqa: BLE001 - a probe must not take the caller down
            logger.warning("verify_point_order probe failed (%s); keeping %s", exc,
                           self.point_field_order)
            return self.point_field_order

        if len(raw) == 0:
            logger.warning("verify_point_order: model pointed nowhere; keeping %s",
                           self.point_field_order)
            return self.point_field_order

        # Whichever of the first two columns is 1 everywhere is the image index.
        col0, col1 = raw[:, 0].astype(int), raw[:, 1].astype(int)
        if (col1 == 1).all() and not (col0 == 1).all():
            order = ("object_id", "image_num", "x", "y")
        elif (col0 == 1).all() and not (col1 == 1).all():
            order = ("image_num", "object_id", "x", "y")
        else:
            logger.warning("verify_point_order inconclusive (cols %s / %s); keeping %s",
                           col0.tolist(), col1.tolist(), self.point_field_order)
            return self.point_field_order

        if order != self.point_field_order:
            logger.warning("point field order is %s, not the configured %s -- switching",
                           order, self.point_field_order)
            self.point_field_order = order
        return order

    def _raw_points(self, images: list[np.ndarray], prompt: str):
        """One request, undecoded, for the order probe."""
        torch = self.torch
        inputs = self.processor.apply_chat_template(
            [self._messages(images, prompt)], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, padding=True, return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        prompt_len = inputs["input_ids"].size(1)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs,
                logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=self.max_new_tokens,
            )
        text = self.processor.post_process_image_text_to_text(
            out[:, prompt_len:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return self.model.extract_image_points(
            text,
            _per_sample(metadata["token_pooling"], 0),
            _per_sample(metadata["subpatch_mapping"], 0),
            _per_sample(metadata["image_sizes"], 0),
        )


def _per_sample(meta, i: int):
    """Slice one sample's entry out of batched pointing metadata.

    The metadata is per-request; whether it arrives as a list or a batched tensor depends
    on the field, so index only when there is something to index.
    """
    if meta is None:
        return None
    if isinstance(meta, (list, tuple)):
        return meta[i]
    if hasattr(meta, "shape") and getattr(meta, "ndim", 0) > 0:
        return meta[i]
    return meta
