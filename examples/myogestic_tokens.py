"""MyoGestic + myoji: classify Rest/Fist from SPAR tokens, live, and drive the Virtual Hand.

MyoGestic's synthetic-EMG classification demo with its RMS/MAV features replaced by
SPAR tokens. Every window becomes one token per electrode, i.e. what a low-bitrate
wearable would transmit, and the classifier sees only the code vectors those tokens
point to.

    uv sync --extra myogestic
    uv run python examples/myogestic_tokens.py

Launch "EMG Generator", record a Rest and a Fist trial, tick both sessions, Train,
then Predict. For a real amplifier, change the stream source, FS and LAYOUT (SPAR needs
the electrode grid geometry).
"""

import sys

import numpy as np
from myogestic import App, Fr, Grid, Stream, TrainingData
from myogestic.controls import ControlLink, load_control_map
from myogestic.ml import Pipeline
from myogestic.ml.widgets import PipelinePanel
from myogestic.recipes.estimators import catboost_classifier
from myogestic.remote import RemoteTarget
from myogestic.session import iter_labeled_windows
from myogestic.sources import LSLSource
from myogestic.tools.emg_generator import control_outlet
from myogestic.vhi import virtual_hand
from myogestic.widgets import (
    PostProcessor,
    PredictionLabel,
    ProcessLauncher,
    RecordingControls,
    SessionManager,
    SignalViewer,
)

from myoji import myogestic as tokens
from myoji.spar import grid_layout

FS = 2048  # the generator below; 2,000 Hz devices need no resampling
LAYOUT = grid_layout([np.arange(8).reshape(2, 4)])  # its 8 channels as a 2x4 grid
VOCABULARY = 64  # the paper's compact codebook; 1,024 codes need far more training windows
PREDICT_HZ = 50  # one SPAR update every 20 ms
HOP_MS = 100
CLASSES = ["Rest", "Fist"]
CONTROL_VALUES = [0.0, 1.0]
PROCESSES = [
    ("EMG Generator", [sys.executable, "-m", "myogestic.tools.emg_generator", "--name", "TestEMG1",
                       "--channels", "8", "--fs", str(FS), "--control", "EMG_Control"]),
]
CONTROL_MAP = load_control_map({"dofs": {"fist": {"targets": [
    {"target": "vhi.prediction.thumb.flexion", "weight": 0.6},
    {"target": "vhi.prediction.index"},
    {"target": "vhi.prediction.middle"},
    {"target": "vhi.prediction.ring"},
    {"target": "vhi.prediction.little"},
], "threshold_fraction": 0.5}}})

vhi = virtual_hand()
vhi_control = vhi.control_client()
generator_control = control_outlet()
output_filter = PostProcessor(hz=PREDICT_HZ)

app = App("MyoGestic with myoji tokens", ui_scale=0.85)
app.streams(Stream("emg", source=LSLSource("TestEMG1"), window_ms=tokens.WINDOW_MS, buffer_ms=60000))
pipeline = Pipeline(app, predict_hz=PREDICT_HZ)
link = ControlLink(CONTROL_MAP, [RemoteTarget(client=vhi_control, interface=vhi)], ctx=app.ctx,
                   smoothing=output_filter, hz=PREDICT_HZ)


@pipeline.extract
def extract(windows: dict[str, np.ndarray]) -> np.ndarray | None:
    """SPAR descriptors of the newest 160 ms, (channels, 18)."""
    window = windows.get("emg")
    return None if window is None else tokens.spar_descriptors(window, FS, LAYOUT)


@pipeline.train
def train(data: TrainingData):
    """Fit the codebook on training windows only, then a classifier on the token vectors."""
    if data.is_empty:
        raise ValueError("No sessions selected.")
    descriptors, labels = [], []
    for window, _ts, class_index in iter_labeled_windows(data.paths, "emg", tokens.WINDOW_MS, HOP_MS,
                                                         classes=data.classes or None):
        descriptors.append(extract({"emg": window}))
        labels.append(class_index)
    if len(set(labels)) < 2:
        raise ValueError("Record trials of both classes.")
    descriptors = np.stack(descriptors)
    codebook = tokens.SparCodebook.fit(descriptors, LAYOUT, vocabulary=VOCABULARY)
    x = codebook.vectors(codebook.tokens(descriptors)).reshape(len(descriptors), -1)
    classifier = catboost_classifier(iterations=100)
    classifier.fit(x, np.array(labels))
    print(f"[train] {len(descriptors)} windows, {VOCABULARY}-code SPAR codebook")
    return codebook, classifier


@pipeline.predict
def predict(model, descriptors: np.ndarray | None) -> dict | None:
    """Quantize, look the codes up, classify, and push the fist activation."""
    if descriptors is None:
        return None
    codebook, classifier = model
    ids = codebook.tokens(descriptors)  # (channels,): the transmitted payload
    proba = classifier.predict_proba(codebook.vectors(ids).reshape(1, -1))[0]
    out = {"class": int(np.argmax(proba)), "proba": proba, "tokens": ids}
    if link.bus is not None:  # never link.ensure() here: it blocks on an RPC
        out["hand"] = link.bus.push({"fist": float(proba[CLASSES.index("Fist")])})
    return out


def on_gesture(index: int) -> None:
    link.ensure()
    generator_control.push_sample(np.array([CONTROL_VALUES[index]], dtype=np.float32))


def on_record() -> None:
    link.ensure()
    app.start_recording()


grid = Grid(7, 3, row_height=[Fr(1)] * 7, col_width=[Fr(1), Fr(1), Fr(1)])
viewer = SignalViewer("emg")
processes = ProcessLauncher(PROCESSES)
recording = RecordingControls(CLASSES, on_record=on_record, on_stop=app.stop_recording, on_gesture=on_gesture)
sessions = SessionManager("sessions", class_names=CLASSES)
panel = PipelinePanel(pipeline)
prediction = PredictionLabel(pipeline, CLASSES)


@app.ui
def layout(ctx):
    with grid[0:7, 1:3]:
        viewer.ui(ctx)
    with grid[0, 0]:
        processes.ui()
    with grid[1, 0]:
        recording.ui(ctx)
    with grid[2:4, 0]:
        pipeline.training_data = sessions.ui()
    with grid[4, 0]:
        panel.ui()
    with grid[5, 0]:
        output_filter.ui()
    with grid[6, 0]:
        prediction.ui()


def main() -> None:
    try:
        app.run()
    finally:
        link.stop()
        vhi_control.stop()


if __name__ == "__main__":
    main()
