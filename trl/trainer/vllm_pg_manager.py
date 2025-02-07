# Disclaimer
# This code is adapted from the following sources:
# - OpenRLHF Pull Request #704: https://github.com/OpenRLHF/OpenRLHF/pull/704
# - AllenAI Open-Instruct Issue #546: https://github.com/allenai/open-instruct/issues/546
# The original code has been modified to suit specific project requirements.
# Please refer to the original repositories for the source implementations.

import torch
from vllm.worker.worker import Worker

class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=True):
        """Use Ray collective process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset

        import ray.util.collective as collective
        collective.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend=backend,
            group_name=group_name
        )
        self._model_update_group = group_name

        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        # Initialize vllm worker with empty tensor
        weight = torch.empty(shape, dtype=dtype, device="cuda")

        import ray.util.collective as collective
        collective.broadcast(weight, 0, group_name=self._model_update_group)
        # The layer names have "model." prepended when getting from self.model_runner.model.named_parameters()
        # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py#L363
        params = list(self.model_runner.model.named_parameters(remove_duplicate=True))
        print("Model parameters:")
        for name, param in params:
            print(f" - {name}")

        loaded_wts = self.model_runner.model.load_weights(weights=[(name, weight)])
        print(f"Loaded tensor to model succesfully: {loaded_wts}")


        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()