from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.server_args import get_global_server_args
from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, make_layers

_is_cuda = is_cuda()


class DotsDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[Any] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # Initialize the parent class first to get all the base functionality
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.speculative_algorithm = get_global_server_args().speculative_algorithm
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=dual_chunk_attention_config,
            alt_stream=alt_stream,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
        )

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        eagle_capture_buff = None
        if getattr(self, "_is_layer_to_capture", False):
            eagle_capture_buff = state.get("eagle_capture_buff")

            # allocate capture buffer
            if eagle_capture_buff is None:
                eagle_capture_buff = []
                state.update(dict(eagle_capture_buff=eagle_capture_buff))

        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=eagle_capture_buff,
            )
        )

        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        expect_keys = {
            "positions",
            "forward_batch",
            "zero_allocator",
            "tbo_subbatch_index",
        }
        for key in expect_keys:
            state.pop(key)

        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )

        # self_attn is DotsAttention/Qwen3MoeAttention, not use zero_allocator
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class DotsModel(DeepseekV2Model):
    def __init__(
        self,
        config,
        quant_config: Optional[Any] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DotsDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0

        for i in range(normal_start_layer, normal_end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        stage_state = None
        if normal_end_layer != self.end_layer:
            import os

            (hidden_states, residual), stage_state = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=not (
                    os.getenv("DISABLE_DOTS_TBO", "0") == "1"
                ),  # if we disable TBO, we can still get overlapped shared experts
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
                require_stage_state=True,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        self.fill_aux_hidden_states(aux_hidden_states, stage_state)
        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    def fill_aux_hidden_states(
        self, aux_hidden_states: List[torch.Tensor], stage_state
    ):
        if isinstance(stage_state, list):
            list_0 = stage_state[0].get("eagle_capture_buff")
            list_1 = stage_state[1].get("eagle_capture_buff")
            if list_0 is not None:
                assert list_1 is not None and len(list_0) == len(list_1)
                for val0, val1 in zip(list_0, list_1):
                    aux_hidden_states.append(torch.cat([val0, val1], dim=0))
            else:
                assert list_1 is None
        elif stage_state is not None:
            val = stage_state.get("eagle_capture_buff")
            if val is not None:
                aux_hidden_states.extend(val)
                stage_state.pop("eagle_capture_buff")


class DotsForCausalLM(DeepseekV2ForCausalLM):
    """Dots model that inherits from DeepseekV2ForCausalLM but uses DotsDecoderLayer"""

    def __init__(
        self,
        config,
        quant_config: Optional[Any] = None,
        prefix: str = "",
    ) -> None:
        from sglang.srt.distributed import (
            get_pp_group,
            get_tensor_model_parallel_world_size,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.utils import add_prefix

        # Initialize basic attributes
        nn.Module.__init__(self)  # Call nn.Module.__init__ directly
        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config

        # Set up fuse_qkv_a_proj from parent class
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping = {
                "fused_qkv_a_proj_with_mqa": [
                    "q_a_proj",
                    "kv_a_proj_with_mqa",
                ]
            }
        else:
            self.packed_modules_mapping = {}

        # Determine num_fused_shared_experts (from parent)
        self.num_fused_shared_experts = 0
        get_global_server_args().disable_shared_experts_fusion = True

        # Create the model with our custom layers
        self.model = DotsModel(config, quant_config, prefix=add_prefix("model", prefix))

        # Create lm_head and logits_processor
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

        # Initialize the lazy value for routed experts weights
        from sglang.srt.models.deepseek_v2 import LazyValue

        def _get_routed_experts():
            result = {}
            if hasattr(self.model, "layers"):
                for layer_id, layer in enumerate(self.model.layers):
                    if hasattr(layer, "mlp"):
                        mlp_layer = getattr(layer, "mlp", None)
                        if mlp_layer is not None and hasattr(
                            mlp_layer, "get_moe_weights"
                        ):
                            try:
                                get_weights_func = getattr(
                                    mlp_layer, "get_moe_weights", None
                                )
                                if callable(get_weights_func):
                                    weights = get_weights_func()
                                    result[layer_id] = weights
                            except (AttributeError, TypeError):
                                continue
            return result

        self._routed_experts_weights_of_layer = LazyValue(_get_routed_experts)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        weights_dict = dict(weights)

        renamed_mappings = [("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")]
        renamed_list = []

        for weight_name, _ in weights_dict.items():
            for old_name, new_name in renamed_mappings:
                if old_name in weight_name:
                    target_name = weight_name.replace(old_name, new_name)
                    renamed_list.append((weight_name, target_name))
        for old_name, new_name in renamed_list:
            weights_dict[new_name] = weights_dict[old_name]
            del weights_dict[old_name]

        extra_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        weights = list(weights_dict.items())

        super().load_weights(
            weights, is_nextn=is_nextn, extra_params_mapping=extra_params_mapping
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self, architecture: str = "DotsForCausalLM"):
        """Inherit the shared expert optimization logic from parent"""
        # For now, disable shared expert fusion for simplicity
        self.num_fused_shared_experts = 0

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_layers_to_capture([val + 1 for val in layer_ids])


# Entry point for the model
EntryClass = DotsForCausalLM
