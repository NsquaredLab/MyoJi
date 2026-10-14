"""Montage-aware channel descriptors for heterogeneous EMG sensors.

An EMG channel is a measurement, not always a single electrode.  Monopolar
channels are represented by the active-electrode centre.  Bipolar and spatial
differential channels are represented by the centre and, when the electrode
endpoints are known, the oriented baseline of the pair.  The acquisition type
is explicit so a downstream spatial readout never has to infer it from signal
amplitude or channel count.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch.nn import functional as F


class MeasurementType(IntEnum):
    """Electrical measurement represented by one input channel."""

    UNKNOWN = 0
    MONOPOLAR = 1
    BIPOLAR = 2
    SPATIAL_DIFFERENTIAL = 3
    INTRAMUSCULAR_BIPOLAR = 4


MEASUREMENT_TYPE_COUNT = len(MeasurementType)
CHANNEL_DESCRIPTOR_DIM = 14


@dataclass(frozen=True)
class ChannelMontage:
    """Geometry and acquisition type for a set of channels.

    ``centres`` and ``directions`` use a sensor-local normalized coordinate
    frame.  ``directions`` point from the negative to the positive endpoint;
    their original norms are stored as ``spacings`` and they are normalized in
    the learned descriptor.  Unknown geometry is represented by zeros plus the
    explicit known/unknown masks, never by invented coordinates.
    """

    centres: torch.Tensor
    measurement_types: torch.Tensor
    directions: torch.Tensor | None = None
    spacings: torch.Tensor | None = None
    coordinate_known: torch.Tensor | None = None
    orientation_known: torch.Tensor | None = None

    def __post_init__(self) -> None:
        centres = torch.as_tensor(self.centres, dtype=torch.float32)
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError("centres must have shape (channel, 3)")
        channels = centres.shape[0]

        measurement_types = torch.as_tensor(self.measurement_types, dtype=torch.long)
        if measurement_types.ndim == 0:
            measurement_types = measurement_types.expand(channels).clone()
        if measurement_types.shape != (channels,):
            raise ValueError("measurement_types must have one entry per channel")
        if torch.any(measurement_types < 0) or torch.any(
            measurement_types >= MEASUREMENT_TYPE_COUNT
        ):
            raise ValueError("measurement_types contains an unknown type id")

        directions = self._channel_matrix(self.directions, channels, "directions")
        if self.spacings is None:
            spacings = torch.linalg.vector_norm(directions, dim=-1)
        else:
            spacings = torch.as_tensor(self.spacings, dtype=torch.float32)
            if spacings.shape != (channels,):
                raise ValueError("spacings must have one entry per channel")
        if torch.any(spacings < 0):
            raise ValueError("spacings cannot be negative")

        coordinate_known = self._channel_mask(
            self.coordinate_known, channels, default=True
        )
        orientation_known = self._channel_mask(
            self.orientation_known,
            channels,
            default=self.directions is not None,
        )
        if not torch.isfinite(centres).all() or not torch.isfinite(directions).all():
            raise ValueError("montage coordinates must be finite")

        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "measurement_types", measurement_types)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "spacings", spacings)
        object.__setattr__(self, "coordinate_known", coordinate_known)
        object.__setattr__(self, "orientation_known", orientation_known)

    @staticmethod
    def _channel_matrix(
        value: torch.Tensor | None, channels: int, name: str
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(channels, 3, dtype=torch.float32)
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.shape != (channels, 3):
            raise ValueError(f"{name} must have shape ({channels}, 3)")
        return value

    @staticmethod
    def _channel_mask(
        value: torch.Tensor | None,
        channels: int,
        default: bool,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((channels,), default, dtype=torch.bool)
        value = torch.as_tensor(value, dtype=torch.bool)
        if value.shape != (channels,):
            raise ValueError("known masks must have one entry per channel")
        return value

    @property
    def channels(self) -> int:
        return self.centres.shape[0]

    def descriptor(self) -> torch.Tensor:
        """Return the fixed-width descriptor consumed after quantization.

        Layout: centre (3), oriented unit baseline (3), ``log1p`` spacing (1),
        coordinate-known (1), orientation-known (1), measurement one-hot (5).
        """

        directions = self.directions
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        unit_directions = torch.where(
            norms > 0,
            directions / norms.clamp_min(torch.finfo(directions.dtype).eps),
            torch.zeros_like(directions),
        )
        unit_directions = unit_directions * self.orientation_known[:, None]
        centres = self.centres * self.coordinate_known[:, None]
        spacings = torch.log1p(self.spacings)[:, None] * self.orientation_known[:, None]
        measurement = F.one_hot(
            self.measurement_types,
            num_classes=MEASUREMENT_TYPE_COUNT,
        ).to(dtype=centres.dtype)
        descriptor = torch.cat(
            (
                centres,
                unit_directions,
                spacings,
                self.coordinate_known[:, None].to(centres.dtype),
                self.orientation_known[:, None].to(centres.dtype),
                measurement,
            ),
            dim=-1,
        )
        if descriptor.shape != (self.channels, CHANNEL_DESCRIPTOR_DIM):
            raise RuntimeError("internal channel descriptor layout is inconsistent")
        return descriptor


def circular_bipolar_montage(channels: int = 16) -> ChannelMontage:
    """Bipolar wrist channels with exact cyclic centres in acquisition order.

    Public EMG2Pose documentation gives channel order around the band but not
    the two physical electrode endpoints for each channel.  Pair direction and
    spacing therefore remain explicitly unknown.
    """

    if channels <= 0:
        raise ValueError("channels must be positive")
    angle = torch.arange(channels, dtype=torch.float32) * (2.0 * torch.pi / channels)
    centres = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), dim=1)
    return ChannelMontage(
        centres=centres,
        measurement_types=torch.full(
            (channels,), int(MeasurementType.BIPOLAR), dtype=torch.long
        ),
        orientation_known=torch.zeros(channels, dtype=torch.bool),
    )


def hd_monopolar_montage() -> ChannelMontage:
    """The 320-channel DPose montage in its five native grid geometries.

    The proximal three grids are 8x8.  The two wrist grids are 13x5 with the
    first 64 of 65 sites recorded.  Coordinates are sensor-local and normalized
    so unequal inter-electrode distances are represented without pretending we
    know a shared 3-D arm registration.
    """

    centres: list[tuple[float, float, float]] = []
    for grid in range(5):
        rows, columns = (8, 8) if grid < 3 else (13, 5)
        sites = 64
        for site in range(sites):
            row, column = divmod(site, columns)
            row_coordinate = 2.0 * row / (rows - 1) - 1.0
            column_coordinate = 2.0 * column / (columns - 1) - 1.0
            grid_coordinate = 2.0 * grid / 4.0 - 1.0
            centres.append((row_coordinate, column_coordinate, grid_coordinate))
    centre_tensor = torch.tensor(centres, dtype=torch.float32)
    return ChannelMontage(
        centres=centre_tensor,
        measurement_types=torch.full(
            (320,), int(MeasurementType.MONOPOLAR), dtype=torch.long
        ),
    )


def grid_monopolar_montage(
    rows: int,
    columns: int,
    channels: int | None = None,
) -> ChannelMontage:
    """Create a normalized row-major descriptor for one surface grid."""

    if rows <= 1 or columns <= 1:
        raise ValueError("grid rows and columns must exceed one")
    sites = rows * columns if channels is None else channels
    if sites <= 0 or sites > rows * columns:
        raise ValueError("channels must fit inside the requested grid")
    centres = []
    for site in range(sites):
        row, column = divmod(site, columns)
        centres.append(
            (
                2.0 * row / (rows - 1) - 1.0,
                2.0 * column / (columns - 1) - 1.0,
                0.0,
            )
        )
    return ChannelMontage(
        centres=torch.tensor(centres, dtype=torch.float32),
        measurement_types=torch.full(
            (sites,), int(MeasurementType.MONOPOLAR), dtype=torch.long
        ),
    )


def unknown_montage(
    channels: int,
    measurement_type: MeasurementType = MeasurementType.UNKNOWN,
) -> ChannelMontage:
    """Describe channels whose physical locations were not recorded."""

    if channels <= 0:
        raise ValueError("channels must be positive")
    return ChannelMontage(
        centres=torch.zeros(channels, 3),
        measurement_types=torch.full(
            (channels,), int(measurement_type), dtype=torch.long
        ),
        coordinate_known=torch.zeros(channels, dtype=torch.bool),
        orientation_known=torch.zeros(channels, dtype=torch.bool),
    )


def spatial_difference_montage(
    monopolar: ChannelMontage,
    negative: torch.Tensor,
    positive: torch.Tensor,
) -> ChannelMontage:
    """Create the oriented-edge montage for ``positive - negative`` channels."""

    negative = torch.as_tensor(negative, dtype=torch.long)
    positive = torch.as_tensor(positive, dtype=torch.long)
    if negative.ndim != 1 or positive.shape != negative.shape:
        raise ValueError("negative and positive must be equally sized index vectors")
    if torch.any(negative < 0) or torch.any(positive < 0):
        raise ValueError("electrode indices cannot be negative")
    if torch.any(negative >= monopolar.channels) or torch.any(
        positive >= monopolar.channels
    ):
        raise ValueError("electrode index is outside the source montage")

    negative_centres = monopolar.centres[negative]
    positive_centres = monopolar.centres[positive]
    directions = positive_centres - negative_centres
    known = (
        monopolar.coordinate_known[negative]
        & monopolar.coordinate_known[positive]
    )
    return ChannelMontage(
        centres=(negative_centres + positive_centres) / 2.0,
        directions=directions,
        spacings=torch.linalg.vector_norm(directions, dim=-1),
        measurement_types=torch.full(
            negative.shape,
            int(MeasurementType.SPATIAL_DIFFERENTIAL),
            dtype=torch.long,
        ),
        coordinate_known=known,
        orientation_known=known,
    )


def apply_spatial_difference(
    emg: torch.Tensor,
    negative: torch.Tensor,
    positive: torch.Tensor,
) -> torch.Tensor:
    """Apply an endpoint-aligned spatial difference on the channel axis."""

    if emg.ndim < 2:
        raise ValueError("EMG must contain channel and time dimensions")
    negative = torch.as_tensor(negative, dtype=torch.long, device=emg.device)
    positive = torch.as_tensor(positive, dtype=torch.long, device=emg.device)
    if negative.shape != positive.shape:
        raise ValueError("negative and positive must have the same shape")
    return emg.index_select(-2, positive) - emg.index_select(-2, negative)
