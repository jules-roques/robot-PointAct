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
    """MolmoPoint-8B, one pointing request per forward (see :meth:`point`)."""

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
        """Point at ``prompts[i]`` within ``image_sets[i]``, one request per forward.

        **Requests are not batched**, and that is not an oversight. The checkpoint's remote
        code cannot run more than one conversation at a time: with two, the pointing
        embed step indexes a per-image tensor with a per-batch-element mask and raises
        ``IndexError: The shape of the mask [2] ... does not match the shape of the indexed
        tensor [4]``. Measured on this checkpoint, all of batch=2 x {1, 2} images fail
        while batch=1 works. Several *images* in one request are fine, which is what
        matters here: the left and right views go in together, so a frame still costs one
        forward rather than two.

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
        return [self._point_one(ims, p) for ims, p in zip(image_sets, prompts)]

    def _point_one(self, images: list[np.ndarray], prompt: str) -> list[Detection]:
        raw = self._raw_points(images, prompt)
        return self._to_detections(raw, n_images=len(images))

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

        Sends a blank image followed by one containing **two** squares. That asymmetry is
        the point: with a single object in the second image both fields read 1 and the
        probe cannot tell them apart (it says so rather than guessing). With two objects in
        image 1, the image index is constant at 1 while the object index varies, so
        whichever column is constant is ``image_num``.

        Returns the field order that holds, and adopts it if it differs from the configured
        one.
        """
        blank = np.full((size, size, 3), 245, dtype=np.uint8)
        two = blank.copy()
        r = size // 10
        for cx in (size // 3, 2 * size // 3):
            two[size // 2 - r:size // 2 + r, cx - r:cx + r] = (200, 30, 30)

        try:
            raw = np.asarray(self._raw_points([blank, two], "Point to the red squares"),
                             dtype=np.float64).reshape(-1, 4)
        except Exception as exc:  # noqa: BLE001 - a probe must not take the caller down
            logger.warning("verify_point_order probe failed (%s); keeping %s", exc,
                           self.point_field_order)
            return self.point_field_order

        col0, col1 = raw[:, 0].astype(int), raw[:, 1].astype(int)
        if len(raw) < 2:
            logger.warning("verify_point_order: got %d point(s), need 2 to disambiguate; "
                           "keeping %s", len(raw), self.point_field_order)
            return self.point_field_order

        # The image index is pinned to 1 (everything visible is in the second image); the
        # object index must vary across the two detections.
        c0_const, c1_const = len(set(col0.tolist())) == 1, len(set(col1.tolist())) == 1
        if c1_const and (col1 == 1).all() and not c0_const:
            order = ("object_id", "image_num", "x", "y")
        elif c0_const and (col0 == 1).all() and not c1_const:
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
        """One conversation in, raw point tuples out.

        The pointing metadata is passed to the decoder whole, exactly as the model card
        does. It describes this single request, and slicing a "per-sample" entry out of it
        produces `ValueError: Mapping and image sizes must have the same length`.
        """
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
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )


