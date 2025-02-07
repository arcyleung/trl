# Disclaimer
# This code is adapted from the following sources:
# - OpenRLHF Pull Request #704: https://github.com/OpenRLHF/OpenRLHF/pull/704
# - AllenAI Open-Instruct Issue #546: https://github.com/allenai/open-instruct/issues/546
# The original code has been modified to suit specific project requirements.
# Please refer to the original repositories for the source implementations.
import os
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote(num_gpus=1, resources={"trl_vllm_engine": 1})
class VLLMEngineActor:
    def __init__(self, *args, **kwargs):
        import vllm
        import os
        print(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.__version__ = vllm.__version__

        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        group_name = kwargs.pop("group_name", "dummy_group")
        trainer_address = kwargs.pop("trainer_address", "127.0.0.1")
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices
        # self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # vLLM engine should join the trainer's cluster
        ray.init(address=trainer_address)

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from vllm_pg_manager import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs["worker_cls"] = "vllm_pg_manager.WorkerWrap"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs["worker_module_name"] = "vllm_pg_manager.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "WorkerWrap"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

        self.init_process_group(
            master_address=trainer_address,
            master_port=29500,
            rank_offset=0,
            world_size=kwargs["tensor_parallel_size"],
            group_name=group_name,
            backend="nccl",
            use_ray=not self.use_gpu_executor
        )

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
            )

    def update_weight(self, layers_and_shapes, dtype, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        for layer, shape in layers_and_shapes.items():
            if self.use_gpu_executor:
                return self.llm.llm_engine.model_executor.driver_worker.update_weight(layer, dtype, shape, empty_cache)
            else:
                return self.llm.llm_engine.model_executor._run_workers("update_weight", layer, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

@ray.remote
def get_all_env_variables():
    import os

    return os.environ


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)

def create_vllm_engines(
    model_path: str,
    num_engines: int,
    tensor_parallel_size: int = 1,
    group_name="dummy_group",
    gpu_memory_utilization=0.9,
    dtype=torch.bfloat16,
    seed=98,
    enable_prefix_caching:bool = True,
    enforce_eager: bool = True,
    max_model_len: int = 32768,

):
    vllm_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    for i in range(num_engines):
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (vLLM mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None

        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        vllm_engines.append(
            VLLMEngineActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model_path,
                noset_visible_devices=noset_visible_devices,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                group_name=group_name,
                dtype=dtype,
                seed=seed + i,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
            )
        )

    return vllm_engines

@ray.remote
class VLLMEngineManager:
    engines = []
    def __init__(
            self,
            model_path,
            num_engines,
            group_name="dummy_group",
            gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            max_model_len=32768,
        ):

        import ray.util.collective as collective

        self.model_path = model_path

        self.engines = create_vllm_engines(
            model_path=self.model_path,
            num_engines=num_engines,
            group_name=group_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=98,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            max_model_len=max_model_len
        )
        print(f"Launched {len(self.engines)} vllm instances of {self.model_path}")

    def update_weights(self, layers_and_shapes):
        for engine in self.engines:
            engine.update_weight.remote(
                layers_and_shapes=layers_and_shapes,
                dtype=self.dtype,
                empty_cache=False
            )

    def generate(self, prompts):
        # Equal balance over each engine instance
        distributed_prompts = [prompts[i::len(self.engines)] for i in range(len(self.engines))]
        generation_tasks = [engine.generate.remote(prompt) for engine, prompt in zip(self.engines, distributed_prompts)]

        # Wait for all engines to finish
        results = ray.get(generation_tasks)

        return results

    def get_engines(self):
        return self.engines

if __name__ == "__main__":
    import torch
    # Testing with Qwen
    manager = VLLMEngineManager.remote(
        model_path="/original_models/Qwen2.5-Coder-3B-Instruct",
        num_engines=2
    )
    completions = ray.get(manager.generate.remote(["San Franciso is a", "Toronto is a", "Kingston is a", "Hong Kong is a"]))

    # print(f"Got named parameters for engine 0 {engines[0].remote().named_parameters.keys()}")

    # update_weight calls ray collective.broadcast inside from rank 0 -> rank1-4
    # This should be called from the trainer side, for each engine, layer by layer

    # i.e. inside the PPO/GRPO trainer update loop, after the weights have been gathered on the rank 0 device:
    # for lyr in self.model_runner.model.named_parameters():
    #     for engine in engines:
    #         engine.update_weight.remote(
    #             name=lyr,
    #             dtype=torch.bfloat16,
    #             shape=(2048),
    #             empty_cache=False
    #         )


    # engines[0].update_weight.remote(
    #     layers_and_shapes= { "model.norm.weight": torch.Size([2048]) },
    #     dtype=torch.bfloat16,
    #     empty_cache=False
    # )

    print(completions)

    while True:
        pass