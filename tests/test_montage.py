import torch

from myoji.learned.montage import (
    CHANNEL_DESCRIPTOR_DIM,
    MeasurementType,
    apply_spatial_difference,
    circular_bipolar_montage,
    hd_monopolar_montage,
    spatial_difference_montage,
    unknown_montage,
)


def test_circular_bipolar_descriptor_is_periodic_and_typed() -> None:
    montage = circular_bipolar_montage(16)
    descriptor = montage.descriptor()

    assert descriptor.shape == (16, CHANNEL_DESCRIPTOR_DIM)
    torch.testing.assert_close(
        torch.linalg.vector_norm(descriptor[:, :2], dim=1),
        torch.ones(16),
    )
    assert torch.all(descriptor[:, 7] == 1)  # centre is known
    assert torch.all(descriptor[:, 8] == 0)  # pair orientation is not published
    assert torch.all(
        montage.measurement_types == int(MeasurementType.BIPOLAR)
    )


def test_hd_montage_uses_native_wrist_grid_shape() -> None:
    montage = hd_monopolar_montage()

    assert montage.channels == 320
    assert torch.unique(montage.centres[:64, 0]).numel() == 8
    assert torch.unique(montage.centres[192:256, 0]).numel() == 13
    assert torch.unique(montage.centres[192:256, 1]).numel() == 5
    assert torch.all(
        montage.measurement_types == int(MeasurementType.MONOPOLAR)
    )


def test_spatial_difference_has_oriented_edge_geometry() -> None:
    source = hd_monopolar_montage()
    negative = torch.tensor([0, 1, 64])
    positive = torch.tensor([8, 9, 72])
    differential = spatial_difference_montage(source, negative, positive)
    emg = torch.arange(2 * 320 * 5).reshape(2, 320, 5)

    expected = emg[:, positive] - emg[:, negative]
    torch.testing.assert_close(
        apply_spatial_difference(emg, negative, positive), expected
    )
    assert differential.descriptor().shape == (3, CHANNEL_DESCRIPTOR_DIM)
    assert torch.all(differential.orientation_known)
    assert torch.all(differential.spacings > 0)
    assert torch.all(
        differential.measurement_types
        == int(MeasurementType.SPATIAL_DIFFERENTIAL)
    )


def test_unknown_geometry_is_not_encoded_as_a_known_origin() -> None:
    montage = unknown_montage(8, MeasurementType.INTRAMUSCULAR_BIPOLAR)
    descriptor = montage.descriptor()

    assert torch.count_nonzero(descriptor[:, :7]) == 0
    assert torch.all(descriptor[:, 7:9] == 0)
    assert torch.all(
        montage.measurement_types == int(MeasurementType.INTRAMUSCULAR_BIPOLAR)
    )
