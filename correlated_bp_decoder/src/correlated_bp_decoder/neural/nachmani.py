"""Torch implementation of the Nachmani-style neural BP decoder."""

from __future__ import annotations

import torch
from torch import nn

from ..math_utils import (
    safe_atanh_exp_signed,
    safe_atanh_exp_signed_,
    safe_log_tanh_split,
    safe_log_tanh_split_,
    sparse_multiply,
    sparse_multiply_,
    xor_affine,
)
from .base import NeuralBPBase


class NachmaniNeuralBP(nn.Module):
    """Nachmani-style unfolded neural belief-propagation decoder.

    Parameters
    ----------
    base
        Compiled graph structure produced by :class:`NeuralBPBase`.
    weights_c2v_v2c
        Optional flattened trainable weights for C2V-to-V2C message edges.
    weights_llrs
        Optional flattened trainable weights for channel-LLR scaling.
    weights_c2v_readout
        Optional trainable weights for the readout edges.
    """

    def __init__(
        self,
        base: NeuralBPBase,
        *,
        weights_c2v_v2c: torch.Tensor | None = None,
        weights_llrs: torch.Tensor | None = None,
        weights_c2v_readout: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base = base

        if weights_c2v_v2c is None:
            weights_c2v_v2c = torch.randn(
                base.nb_weights_c2v_v2c * base.n_layers,
                dtype=torch.float32,
            )
        if weights_llrs is None:
            weights_llrs = torch.randn(
                base.code_n_bits * base.n_layers,
                dtype=torch.float32,
            )
        if weights_c2v_readout is None:
            weights_c2v_readout = torch.randn(
                base.nb_weights_c2v_readout,
                dtype=torch.float32,
            )

        self.weights_c2v_v2c = nn.Parameter(weights_c2v_v2c.reshape(-1).to(torch.float32))
        self.weights_llrs = nn.Parameter(weights_llrs.reshape(-1).to(torch.float32))
        self.weights_c2v_readout = nn.Parameter(
            weights_c2v_readout.reshape(-1).to(torch.float32)
        )

        self.register_buffer(
            "adj_v2c_c2v_bool",
            torch.as_tensor(base.adj_v2c_c2v.astype("bool")),
            persistent=False,
        )
        self.register_buffer(
            "adj_v2c_c2v_float",
            torch.as_tensor(base.adj_v2c_c2v.astype("float32")),
            persistent=False,
        )
        self.register_buffer(
            "edge_to_checks",
            torch.as_tensor(base.edge_to_checks, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "non_zero_rows_c2v_v2c",
            torch.as_tensor(base.non_zero_rows_c2v_v2c, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "non_zero_cols_c2v_v2c",
            torch.as_tensor(base.non_zero_cols_c2v_v2c, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "non_zero_rows_c2v_readout",
            torch.as_tensor(base.non_zero_rows_c2v_readout, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "non_zero_cols_c2v_readout",
            torch.as_tensor(base.non_zero_cols_c2v_readout, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "adj_initialize_v2c_float",
            torch.as_tensor(base.adj_initialize_v2c.astype("float32")),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the device currently holding the trainable parameters."""

        return self.weights_llrs.device

    def prepare_inputs(
        self,
        initial_llrs_batch: torch.Tensor | object,
        syndromes_batch: torch.Tensor | object,
        *,
        batch_first: bool = False,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize forward inputs to the model's internal layout and device.

        Parameters
        ----------
        initial_llrs_batch
            Channel-LLR batch. Expected shape is ``(n_bits, n_samples)`` unless
            ``batch_first`` is ``True``, in which case ``(n_samples, n_bits)``
            is accepted.
        syndromes_batch
            Syndrome batch. Expected shape is ``(n_checks, n_samples)`` unless
            ``batch_first`` is ``True``, in which case ``(n_samples, n_checks)``
            is accepted.
        batch_first
            Whether the provided inputs use batch-first layout.
        device
            Optional target device. Defaults to the model's current device.

        Returns
        -------
        tuple
            Normalized ``(initial_llrs_batch, syndromes_batch)`` tensors in the
            internal ``(features, samples)`` layout.
        """

        target_device = self.device if device is None else torch.device(device)
        initial_llrs_tensor = torch.as_tensor(
            initial_llrs_batch,
            dtype=torch.float32,
            device=target_device,
        )
        syndromes_tensor = torch.as_tensor(
            syndromes_batch,
            dtype=torch.bool,
            device=target_device,
        )
        if batch_first:
            if initial_llrs_tensor.ndim != 2:
                raise ValueError(
                    "batch-first initial_llrs_batch must have shape "
                    "(n_samples, n_bits)."
                )
            if syndromes_tensor.ndim != 2:
                raise ValueError(
                    "batch-first syndromes_batch must have shape "
                    "(n_samples, n_checks)."
                )
            initial_llrs_tensor = initial_llrs_tensor.transpose(0, 1).contiguous()
            syndromes_tensor = syndromes_tensor.transpose(0, 1).contiguous()
        return _validate_forward_inputs(
            self.base,
            initial_llrs_tensor,
            syndromes_tensor,
        )

    def expand_initial_llrs(
        self,
        n_samples: int,
        *,
        batch_first: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Broadcast the base channel LLR prior across a batch.

        Parameters
        ----------
        n_samples
            Number of samples in the batch.
        batch_first
            Whether to return ``(n_samples, n_bits)`` instead of the internal
            ``(n_bits, n_samples)`` layout.
        device
            Optional target device. Defaults to the model's current device.

        Returns
        -------
        torch.Tensor
            Repeated initial channel LLRs.
        """

        target_device = self.device if device is None else torch.device(device)
        llrs = torch.as_tensor(
            self.base.initial_llrs,
            dtype=torch.float32,
            device=target_device,
        ).unsqueeze(1).repeat(1, n_samples)
        if batch_first:
            return llrs.transpose(0, 1).contiguous()
        return llrs

    def forward(
        self,
        initial_llrs_batch: torch.Tensor | object,
        syndromes_batch: torch.Tensor | object,
        *,
        batch_first: bool = False,
        return_batch_first: bool | None = None,
    ) -> torch.Tensor:
        """Run the in-place forward pass and return per-layer posterior LLRs.

        Parameters
        ----------
        initial_llrs_batch
            Channel-LLR batch. Expected shape is ``(n_bits, n_samples)`` unless
            ``batch_first`` is ``True``.
        syndromes_batch
            Syndrome batch. Expected shape is ``(n_checks, n_samples)`` unless
            ``batch_first`` is ``True``.
        batch_first
            Whether the provided inputs use batch-first layout.
        return_batch_first
            Whether to return ``(n_samples, n_bits, n_layers)``. When omitted,
            it defaults to the same value as ``batch_first``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_bits, n_samples, n_layers)`` by default, or
            ``(n_samples, n_bits, n_layers)`` when ``return_batch_first`` is
            requested.
        """

        initial_llrs_batch, syndromes_batch = self.prepare_inputs(
            initial_llrs_batch,
            syndromes_batch,
            batch_first=batch_first,
        )
        posterior_llrs = forward_pass_with_weights(
            self.weights_c2v_v2c,
            self.weights_llrs,
            self.weights_c2v_readout,
            self.base,
            initial_llrs_batch,
            syndromes_batch,
            module=self,
        )
        if return_batch_first is None:
            return_batch_first = batch_first
        if return_batch_first:
            return posterior_llrs.permute(1, 0, 2).contiguous()
        return posterior_llrs

    def forward_batch_first(
        self,
        initial_llrs_batch: torch.Tensor | object,
        syndromes_batch: torch.Tensor | object,
    ) -> torch.Tensor:
        """Run the forward pass using batch-first inputs and outputs.

        Parameters
        ----------
        initial_llrs_batch
            Tensor-like batch of shape ``(n_samples, n_bits)``.
        syndromes_batch
            Tensor-like batch of shape ``(n_samples, n_checks)``.

        Returns
        -------
        torch.Tensor
            Posterior LLRs with shape ``(n_samples, n_bits, n_layers)``.
        """

        return self(
            initial_llrs_batch,
            syndromes_batch,
            batch_first=True,
            return_batch_first=True,
        )


def c2v_to_v2c(
    messages_c2v_previous: torch.Tensor,
    weights_llrs: torch.Tensor,
    weights_messages: torch.Tensor,
    channel_llrs: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional C2V-to-V2C update.

    Parameters
    ----------
    messages_c2v_previous
        Previous C2V messages with shape ``(n_edges, n_samples)``.
    weights_llrs
        Per-bit channel weights for one layer.
    weights_messages
        Per-edge message weights for one layer.
    channel_llrs
        Channel LLRs with shape ``(n_bits, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers for the compiled graph.

    Returns
    -------
    magnitudes, signs
        Activated V2C message representation.
    """

    rows, cols, _, _, adj_initialize = _graph_tensors(base, channel_llrs, module)
    messages_v2c = sparse_multiply(
        rows,
        cols,
        weights_messages.to(channel_llrs.dtype),
        messages_c2v_previous,
        n_rows=base.nb_neurons_per_layer,
    )
    messages_v2c = messages_v2c + adj_initialize @ (weights_llrs.unsqueeze(1) * channel_llrs)
    return safe_log_tanh_split(messages_v2c)


def c2v_to_v2c_with_weights_(
    activated_m_v2c_magnitudes: torch.Tensor,
    activated_m_v2c_signs: torch.Tensor,
    messages_v2c: torch.Tensor,
    messages_c2v: torch.Tensor,
    weighted_channel_llrs: torch.Tensor,
    weights_llr_layer: torch.Tensor,
    weights_c2v_v2c_layer: torch.Tensor,
    initial_llrs_batch: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> None:
    """In-place C2V-to-V2C update.

    Parameters
    ----------
    activated_m_v2c_magnitudes
        Output tensor receiving activated magnitudes.
    activated_m_v2c_signs
        Output tensor receiving activated signs.
    messages_v2c
        Preallocated V2C message tensor.
    messages_c2v
        Input C2V message tensor from the previous layer.
    weighted_channel_llrs
        Preallocated buffer for weighted channel LLRs.
    weights_llr_layer
        Per-bit channel weights for one layer.
    weights_c2v_v2c_layer
        Per-edge message weights for one layer.
    initial_llrs_batch
        Channel LLRs with shape ``(n_bits, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers.
    """

    rows, cols, _, _, adj_initialize = _graph_tensors(base, initial_llrs_batch, module)
    sparse_multiply_(
        messages_v2c,
        rows,
        cols,
        weights_c2v_v2c_layer.to(initial_llrs_batch.dtype),
        messages_c2v,
    )
    weighted_channel_llrs.copy_(weights_llr_layer.unsqueeze(1) * initial_llrs_batch)
    messages_v2c.add_(adj_initialize @ weighted_channel_llrs)
    safe_log_tanh_split_(activated_m_v2c_magnitudes, activated_m_v2c_signs, messages_v2c)


def v2c_to_c2v(
    activated_m_v2c_magnitudes: torch.Tensor,
    activated_m_v2c_signs: torch.Tensor,
    syndromes_batch: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> torch.Tensor:
    """Functional V2C-to-C2V update.

    Parameters
    ----------
    activated_m_v2c_magnitudes
        Activated V2C magnitudes.
    activated_m_v2c_signs
        Activated V2C signs.
    syndromes_batch
        Syndrome tensor with shape ``(n_checks, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers.

    Returns
    -------
    torch.Tensor
        C2V message tensor with shape ``(n_edges, n_samples)``.
    """

    _, _, adj_v2c_c2v, edge_to_checks, _ = _graph_tensors(base, activated_m_v2c_magnitudes, module)
    activated_m_c2v_magnitudes = adj_v2c_c2v @ activated_m_v2c_magnitudes
    syndrome_signs = syndromes_batch.index_select(0, edge_to_checks).to(torch.bool)
    activated_m_c2v_signs = xor_affine(
        _graph_bool_adjacency(base, activated_m_v2c_magnitudes.device, module),
        activated_m_v2c_signs,
        syndrome_signs,
    )
    return safe_atanh_exp_signed(activated_m_c2v_magnitudes, activated_m_c2v_signs)


def v2c_to_c2v_(
    messages_c2v: torch.Tensor,
    activated_m_c2v_magnitudes: torch.Tensor,
    activated_m_c2v_signs: torch.Tensor,
    activated_m_v2c_magnitudes: torch.Tensor,
    activated_m_v2c_signs: torch.Tensor,
    syndromes_batch: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> None:
    """In-place V2C-to-C2V update.

    Parameters
    ----------
    messages_c2v
        Output tensor receiving C2V messages.
    activated_m_c2v_magnitudes
        Buffer receiving aggregated magnitudes.
    activated_m_c2v_signs
        Buffer receiving aggregated signs.
    activated_m_v2c_magnitudes
        Activated V2C magnitudes.
    activated_m_v2c_signs
        Activated V2C signs.
    syndromes_batch
        Syndrome tensor with shape ``(n_checks, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers.
    """

    _, _, adj_v2c_c2v, edge_to_checks, _ = _graph_tensors(base, activated_m_v2c_magnitudes, module)
    activated_m_c2v_magnitudes.copy_(adj_v2c_c2v @ activated_m_v2c_magnitudes)
    syndrome_signs = syndromes_batch.index_select(0, edge_to_checks).to(torch.bool)
    activated_m_c2v_signs.copy_(
        xor_affine(
            _graph_bool_adjacency(base, activated_m_v2c_magnitudes.device, module),
            activated_m_v2c_signs,
            syndrome_signs,
        )
    )
    safe_atanh_exp_signed_(messages_c2v, activated_m_c2v_magnitudes, activated_m_c2v_signs)


def readout(
    messages_c2v: torch.Tensor,
    weights_readout: torch.Tensor,
    weights_llrs: torch.Tensor,
    channel_llrs: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> torch.Tensor:
    """Functional posterior readout from C2V messages.

    Parameters
    ----------
    messages_c2v
        C2V message tensor with shape ``(n_edges, n_samples)``.
    weights_readout
        Readout-edge weights.
    weights_llrs
        Per-bit channel weights for one layer.
    channel_llrs
        Channel LLRs with shape ``(n_bits, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers.

    Returns
    -------
    torch.Tensor
        Posterior LLRs with shape ``(n_bits, n_samples)``.
    """

    _, _, _, _, _ = _graph_tensors(base, channel_llrs, module)
    if module is not None:
        rows = module.non_zero_rows_c2v_readout
        cols = module.non_zero_cols_c2v_readout
    else:
        rows = torch.as_tensor(base.non_zero_rows_c2v_readout, dtype=torch.long, device=channel_llrs.device)
        cols = torch.as_tensor(base.non_zero_cols_c2v_readout, dtype=torch.long, device=channel_llrs.device)
    posterior_llrs = sparse_multiply(
        rows,
        cols,
        weights_readout.to(channel_llrs.dtype),
        messages_c2v,
        n_rows=base.code_n_bits,
    )
    return posterior_llrs + weights_llrs.unsqueeze(1) * channel_llrs


def readout_with_weights_(
    posterior_llrs: torch.Tensor,
    messages_c2v: torch.Tensor,
    weights_readout: torch.Tensor,
    weights_llrs: torch.Tensor,
    channel_llrs: torch.Tensor,
    base: NeuralBPBase,
    *,
    module: NachmaniNeuralBP | None = None,
) -> None:
    """In-place posterior readout from C2V messages.

    Parameters
    ----------
    posterior_llrs
        Output tensor receiving posterior LLRs.
    messages_c2v
        C2V message tensor with shape ``(n_edges, n_samples)``.
    weights_readout
        Readout-edge weights.
    weights_llrs
        Per-bit channel weights for one layer.
    channel_llrs
        Channel LLRs with shape ``(n_bits, n_samples)``.
    base
        Neural BP graph structure.
    module
        Optional module providing cached torch buffers.
    """

    if module is not None:
        rows = module.non_zero_rows_c2v_readout
        cols = module.non_zero_cols_c2v_readout
    else:
        rows = torch.as_tensor(base.non_zero_rows_c2v_readout, dtype=torch.long, device=channel_llrs.device)
        cols = torch.as_tensor(base.non_zero_cols_c2v_readout, dtype=torch.long, device=channel_llrs.device)
    sparse_multiply_(
        posterior_llrs,
        rows,
        cols,
        weights_readout.to(channel_llrs.dtype),
        messages_c2v,
    )
    posterior_llrs.add_(weights_llrs.unsqueeze(1) * channel_llrs)


def get_layer_weights(
    weights_c2v_v2c: torch.Tensor,
    weights_llrs: torch.Tensor,
    base: NeuralBPBase,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the weights associated with a single unfolded BP layer.

    Parameters
    ----------
    weights_c2v_v2c
        Flattened message-edge weights across all layers.
    weights_llrs
        Flattened channel weights across all layers.
    base
        Neural BP graph structure.
    layer
        Zero-based layer index.

    Returns
    -------
    weights_messages, weights_llr_layer
        The per-layer message and channel weights.
    """

    start = layer * base.nb_weights_c2v_v2c
    end = (layer + 1) * base.nb_weights_c2v_v2c
    llr_start = layer * base.code_n_bits
    llr_end = (layer + 1) * base.code_n_bits
    return weights_c2v_v2c[start:end], weights_llrs[llr_start:llr_end]


def forward_pass_with_weights(
    weights_c2v_v2c: torch.Tensor,
    weights_llrs: torch.Tensor,
    weights_c2v_readout: torch.Tensor,
    base: NeuralBPBase,
    initial_llrs_batch: torch.Tensor,
    syndromes_batch: torch.Tensor,
    *,
    module: NachmaniNeuralBP | None = None,
) -> torch.Tensor:
    """Run the explicit-weight forward pass through the unfolded network.

    Parameters
    ----------
    weights_c2v_v2c
        Flattened message-edge weights across all layers.
    weights_llrs
        Flattened channel weights across all layers.
    weights_c2v_readout
        Readout-edge weights.
    base
        Neural BP graph structure.
    initial_llrs_batch
        Tensor of shape ``(n_bits, n_samples)``.
    syndromes_batch
        Tensor of shape ``(n_checks, n_samples)``.
    module
        Optional module providing cached graph buffers.

    Returns
    -------
    torch.Tensor
        Posterior LLRs of shape ``(n_bits, n_samples, n_layers)``.
    """

    initial_llrs_batch, syndromes_batch = _validate_forward_inputs(
        base,
        initial_llrs_batch,
        syndromes_batch,
    )
    dtype = initial_llrs_batch.dtype
    device = initial_llrs_batch.device
    n_samples = initial_llrs_batch.shape[1]
    n_edges = base.nb_neurons_per_layer

    messages_c2v = torch.zeros((n_edges, n_samples), dtype=dtype, device=device)
    posterior_layers: list[torch.Tensor] = []

    for layer in range(base.n_layers):
        weights_messages, weights_llr_layer = get_layer_weights(
            weights_c2v_v2c,
            weights_llrs,
            base,
            layer,
        )
        activated_m_v2c_magnitudes, activated_m_v2c_signs = c2v_to_v2c(
            messages_c2v,
            weights_llr_layer,
            weights_messages,
            initial_llrs_batch,
            base,
            module=module,
        )
        messages_c2v = v2c_to_c2v(
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            syndromes_batch,
            base,
            module=module,
        )
        posterior_llrs_layer = readout(
            messages_c2v,
            weights_c2v_readout,
            weights_llr_layer,
            initial_llrs_batch,
            base,
            module=module,
        )
        posterior_layers.append(posterior_llrs_layer)

    return torch.stack(posterior_layers, dim=2)


def _graph_tensors(
    base: NeuralBPBase,
    reference: torch.Tensor,
    module: NachmaniNeuralBP | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the torch graph tensors on the correct device.

    Parameters
    ----------
    base
        Neural BP graph structure.
    reference
        Tensor providing the target device and dtype context.
    module
        Optional module containing cached buffers.

    Returns
    -------
    tuple
        ``(rows_c2v_v2c, cols_c2v_v2c, adj_v2c_c2v, edge_to_checks, adj_initialize)``.
    """

    if module is not None:
        return (
            module.non_zero_rows_c2v_v2c,
            module.non_zero_cols_c2v_v2c,
            module.adj_v2c_c2v_float,
            module.edge_to_checks,
            module.adj_initialize_v2c_float,
        )

    rows = torch.as_tensor(base.non_zero_rows_c2v_v2c, dtype=torch.long, device=reference.device)
    cols = torch.as_tensor(base.non_zero_cols_c2v_v2c, dtype=torch.long, device=reference.device)
    adj_v2c_c2v = torch.as_tensor(
        base.adj_v2c_c2v.astype("float32"),
        dtype=reference.dtype,
        device=reference.device,
    )
    edge_to_checks = torch.as_tensor(base.edge_to_checks, dtype=torch.long, device=reference.device)
    adj_initialize = torch.as_tensor(
        base.adj_initialize_v2c.astype("float32"),
        dtype=reference.dtype,
        device=reference.device,
    )
    return rows, cols, adj_v2c_c2v, edge_to_checks, adj_initialize


def _graph_bool_adjacency(
    base: NeuralBPBase,
    device: torch.device,
    module: NachmaniNeuralBP | None,
) -> torch.Tensor:
    """Return the boolean V2C-to-C2V adjacency tensor on the target device.

    Parameters
    ----------
    base
        Neural BP graph structure.
    device
        Target device for the boolean adjacency.
    module
        Optional module providing cached buffers.

    Returns
    -------
    torch.Tensor
        Boolean adjacency tensor.
    """

    if module is not None:
        return module.adj_v2c_c2v_bool
    return torch.as_tensor(base.adj_v2c_c2v.astype("bool"), dtype=torch.bool, device=device)


def _validate_forward_inputs(
    base: NeuralBPBase,
    initial_llrs_batch: torch.Tensor,
    syndromes_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize forward-pass input tensors.

    Parameters
    ----------
    base
        Neural BP graph structure.
    initial_llrs_batch
        Candidate channel-LLR tensor.
    syndromes_batch
        Candidate syndrome tensor.

    Returns
    -------
    tuple
        Normalized ``float32``/``bool`` tensors.
    """

    if initial_llrs_batch.ndim != 2:
        raise ValueError("initial_llrs_batch must have shape (n_bits, n_samples).")
    if initial_llrs_batch.shape[0] != base.code_n_bits:
        raise ValueError("initial_llrs_batch has the wrong number of bit rows.")
    if syndromes_batch.ndim != 2:
        raise ValueError("syndromes_batch must have shape (n_checks, n_samples).")
    if syndromes_batch.shape[0] != base.code_n_checks:
        raise ValueError("syndromes_batch has the wrong number of check rows.")
    if syndromes_batch.shape[1] != initial_llrs_batch.shape[1]:
        raise ValueError("initial_llrs_batch and syndromes_batch must share the sample axis.")
    return initial_llrs_batch.to(torch.float32), syndromes_batch.to(torch.bool)
