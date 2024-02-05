from pprint import pprint

from generative_models.sgm.inference.api import (
    SamplingPipeline,
    SamplingSpec,
    ModelArchitecture,
)
import generative_models.sgm.inference.helpers as helpers

def main():
    model_specs = {
        ModelArchitecture.SDXL_V1_BASE: SamplingSpec(
            height=1024,
            width=1024,
            channels=4,
            factor=8,
            is_legacy=False,
            config="sd_xl_base.yaml",
            ckpt="sd_xl_base_1.0.safetensors",
            is_guided=True,
        ),
        ModelArchitecture.SDXL_V1_REFINER: SamplingSpec(
            height=1024,
            width=1024,
            channels=4,
            factor=8,
            is_legacy=True,
            config="sd_xl_refiner.yaml",
            ckpt="sd_xl_refiner_1.0.safetensors",
            is_guided=True,
        )
    }

    params = [ModelArchitecture.SDXL_V1_BASE, ModelArchitecture.SDXL_V1_REFINER]
    base_pipeline = SamplingPipeline(params[0], config_path="generative_models/configs/inference")

    pprint(vars(base_pipeline.model))


if __name__ == '__main__':
    main()