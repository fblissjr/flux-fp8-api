# modules/clip_slider.py

import torch
from tqdm import tqdm
import random
from typing import Optional, Tuple, List
from loguru import logger


class CLIPSliderFlux:
    def __init__(self, pipe, device: torch.device):
        self.pipe = pipe
        self.device = device

    def interpolate_prompts(
        self, prompt1: str, prompt2: str, num_steps: int
    ) -> List[torch.Tensor]:
        """
        Interpolate between two prompts in the embedding space.

        Args:
            prompt1 (str): The starting prompt.
            prompt2 (str): The ending prompt.
            num_steps (int): Number of interpolation steps.

        Returns:
            List[torch.Tensor]: List of interpolated embeddings.
        """
        try:
            clip_embeds1, _, _ = self.pipe.get_text_embeddings(prompt1)
            clip_embeds2, _, _ = self.pipe.get_text_embeddings(prompt2)

            interpolated_embeds = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interpolated_embed = (1 - alpha) * clip_embeds1 + alpha * clip_embeds2
                interpolated_embeds.append(interpolated_embed)

            return interpolated_embeds
        except Exception as e:
            logger.error(f"Error in interpolate_prompts: {str(e)}")
            raise RuntimeError("Failed to interpolate prompts") from e

    def generate_interpolation(
        self,
        prompt1: str,
        prompt2: str,
        num_steps: int,
        width: int,
        height: int,
        guidance: float,
        seed: int,
        inference_steps: int,
    ) -> List[bytes]:
        """
        Generate a sequence of images interpolating between two prompts.

        Args:
            prompt1 (str): The starting prompt.
            prompt2 (str): The ending prompt.
            num_steps (int): Number of interpolation steps.
            width (int): Width of the generated images.
            height (int): Height of the generated images.
            guidance (float): Guidance scale for generation.
            seed (int): Random seed for reproducibility.
            inference_steps (int): Number of denoising steps for each image.

        Returns:
            List[bytes]: List of generated images in bytes format.
        """
        try:
            interpolated_embeds = self.interpolate_prompts(prompt1, prompt2, num_steps)
            generator, seed = self.pipe.set_seed(seed)

            images = []
            for i, embed in enumerate(interpolated_embeds):
                logger.info(f"Generating image {i+1}/{num_steps}")

                img, timesteps = self.pipe.preprocess_latent(
                    init_image=None,
                    height=height,
                    width=width,
                    num_steps=inference_steps,
                    strength=1.0,
                    generator=generator,
                    num_images=1,
                )

                _, img_ids, _, txt, txt_ids = self.pipe.prepare(
                    img=img,
                    prompt=prompt1,  # We use prompt1 here, but it doesn't matter as we'll override the embeddings
                    target_device=self.pipe.device_flux,
                    target_dtype=self.pipe.dtype,
                )

                guidance_vec = torch.full(
                    (img.shape[0],),
                    guidance,
                    device=self.pipe.device_flux,
                    dtype=self.pipe.dtype,
                )

                for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                    t_vec = torch.full(
                        (img.shape[0],),
                        t_curr,
                        dtype=self.pipe.dtype,
                        device=self.pipe.device_flux,
                    )

                    pred = self.pipe.model.forward(
                        img=img,
                        img_ids=img_ids,
                        txt=txt,
                        txt_ids=txt_ids,
                        y=embed,  # Use the interpolated embedding
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )

                    img = img + (t_prev - t_curr) * pred

                img = self.pipe.vae_decode(img, height, width)
                images.append(self.pipe.into_bytes(img))

            return images
        except Exception as e:
            logger.error(f"Error in generate_interpolation: {str(e)}")
            raise RuntimeError("Failed to generate interpolation") from e

    def find_latent_direction(
        self, target_word: str, opposite: str, num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calculate the latent direction between two concepts.

        Args:
            target_word (str): The target concept.
            opposite (str): The opposite concept.
            num_iterations (Optional[int]): Number of iterations for calculation. If None, uses self.iterations.

        Returns:
            torch.Tensor: The calculated latent direction.

        Raises:
            RuntimeError: If there's an issue with embedding generation.
        """
        iterations = num_iterations or self.iterations

        try:
            with torch.no_grad():
                positives = []
                negatives = []
                for _ in tqdm(range(iterations), desc="Calculating latent direction"):
                    medium = random.choice(MEDIUMS)
                    subject = random.choice(SUBJECTS)
                    pos_prompt = f"a {medium} of a {target_word} {subject}"
                    neg_prompt = f"a {medium} of a {opposite} {subject}"

                    pos_clip_embeds, _, _ = self.pipe.get_text_embeddings(pos_prompt)
                    neg_clip_embeds, _, _ = self.pipe.get_text_embeddings(neg_prompt)

                    positives.append(pos_clip_embeds)
                    negatives.append(neg_clip_embeds)

            positives = torch.cat(positives, dim=0)
            negatives = torch.cat(negatives, dim=0)
            diffs = positives - negatives
            return diffs.mean(0, keepdim=True)
        except Exception as e:
            logger.error(f"Error in find_latent_direction: {str(e)}")
            raise RuntimeError("Failed to calculate latent direction") from e

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        scale: float,
        seed: int,
        num_steps: int,
        avg_diff: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bytes:
        """
        Generate an image based on the given parameters.

        Args:
            prompt (str): The text prompt for image generation.
            width (int): Width of the generated image.
            height (int): Height of the generated image.
            guidance (float): Guidance scale for generation.
            scale (float): Scale factor for latent direction.
            seed (int): Random seed for reproducibility.
            num_steps (int): Number of denoising steps.
            avg_diff (Optional[torch.Tensor]): Pre-calculated latent direction.

        Returns:
            bytes: The generated image in bytes format.

        Raises:
            RuntimeError: If there's an issue during image generation.
        """
        try:
            generator, seed = self.pipe.set_seed(seed)

            img, timesteps = self.pipe.preprocess_latent(
                init_image=None,
                height=height,
                width=width,
                num_steps=num_steps,
                strength=1.0,
                generator=generator,
                num_images=1,
            )

            img, img_ids, vec, txt, txt_ids = map(
                lambda x: x.contiguous(),
                self.pipe.prepare(
                    img=img,
                    prompt=prompt,
                    target_device=self.pipe.device_flux,
                    target_dtype=self.pipe.dtype,
                ),
            )

            if avg_diff is not None:
                vec = vec + avg_diff * scale

            guidance_vec = torch.full(
                (img.shape[0],),
                guidance,
                device=self.pipe.device_flux,
                dtype=self.pipe.dtype,
            )

            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full(
                    (img.shape[0],),
                    t_curr,
                    dtype=self.pipe.dtype,
                    device=self.pipe.device_flux,
                )

                pred = self.pipe.model.forward(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                img = img + (t_prev - t_curr) * pred

            img = self.pipe.vae_decode(img, height, width)
            return self.pipe.into_bytes(img)
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise RuntimeError("Failed to generate image") from e
