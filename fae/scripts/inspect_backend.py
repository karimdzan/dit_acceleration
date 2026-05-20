import hydra
from omegaconf import DictConfig, OmegaConf

from fae.generators.common import LatentTensorSpec
from fae.scripts.common import (
    build_bridge_from_config,
    build_device,
    build_generator_from_config,
    count_parameters,
    count_trainable_parameters,
    get_fae_latent_spec,
    build_backbone_from_config,
    build_fae_from_config,
)


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) :
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    device = build_device(config)

    backbone = build_backbone_from_config(config).to(device)
    autoencoder = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    ae_spec = get_fae_latent_spec(config, input_dim=backbone.output_dim, backbone=backbone)

    generator_spec = LatentTensorSpec(
        channels=config['generator'].get('in_channels', ae_spec.channels),
        height=config['generator'].get('sample_size', ae_spec.height),
        width=config['generator'].get('sample_size', ae_spec.width),
    )
    generator = build_generator_from_config(config, model_spec=generator_spec).to(device)
    bridge = build_bridge_from_config(config, fae_spec=ae_spec, model_spec=generator.latent_spec()).to(device)

    print(f'generator={config["generator"]["name"]}')
    print(f'device={device}')
    print(f'ae_spec={ae_spec}')
    print(f'generator_spec={generator.latent_spec()}')
    print(f'autoencoder_params={count_parameters(autoencoder):,}')
    print(f'generator_params={count_parameters(generator):,}')
    print(f'generator_trainable={count_trainable_parameters(generator):,}')
    print(f'bridge_params={count_parameters(bridge):,}')
    print(f'bridge_trainable={count_trainable_parameters(bridge):,}')


if __name__ == "__main__":
    main()
