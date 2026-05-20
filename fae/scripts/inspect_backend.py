import hydra
from omegaconf import DictConfig, OmegaConf

from fae.generators.common import LatentTensorSpec
from fae.scripts.common import build_bridge_from_config, build_device, build_generator_from_config, count_parameters, count_trainable_parameters


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    device = build_device(config)
    seed_spec = LatentTensorSpec(
        channels=config['generator'].get('in_channels', config['fae'].get('latent_dim', 32)),
        height=config['generator'].get('sample_size', 16),
        width=config['generator'].get('sample_size', 16),
    )
    backend = build_generator_from_config(config, model_spec=seed_spec).to(device)
    bridge = build_bridge_from_config(config, backend.latent_spec()).to(device)

    print(f'backend={config["generator"]["name"]}')
    print(f'device={device}')
    print(f'latent_spec={backend.latent_spec()}')
    print(f'backend_params={count_parameters(backend):,}')
    print(f'backend_trainable={count_trainable_parameters(backend):,}')
    print(f'bridge_params={count_parameters(bridge):,}')
    print(f'bridge_trainable={count_trainable_parameters(bridge):,}')


if __name__ == "__main__":
    main()
