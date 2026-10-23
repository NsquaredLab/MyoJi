# Vendored code

These files are copied **byte-for-byte** (scripts pin their SHA-256) from
[MyoTorch](https://github.com/NsquaredLab/MyoTorch) (formerly MyoVerse), by Raul C. Sîmpetru and
Vlad Cnejevici, licensed under the GNU General Public License v3.0 (`LICENSE-GPL-3.0` here).
They are combined with the AGPL-3.0-or-later code of this project as permitted by section 13 of
both licences; these files themselves remain under the GPL-3.0.

| file | contents |
|---|---|
| `channel_tokenization.py` | `ChannelPatchEncoder`, `GeometryAwareChannelPool`, `ChannelTokenizerFrontend` |
| `raulnet_v20.py` | `RaulNetV20`: the spatial CNN + TCN pose decoder over token or feature frames |
