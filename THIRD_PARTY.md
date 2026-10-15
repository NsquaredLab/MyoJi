# Third-party code and data

myoji is licensed under the GNU Affero General Public License v3.0 or later (`LICENSE`).
The following parts come from elsewhere and keep their own terms.

## Included

| what | where | origin | licence |
|---|---|---|---|
| SPAR feature extraction | `src/myoji/spar.py` | Adapted from MyoKit's `SPAR_RIE_M` / `SPAR_SIMPLE` by Amin Olamazadeh (commit `e19a2fcf8dbbcf836e586eadc7dd9dec2942d252`) | as this repository |
| Channel tokenization frontend, RaulNet V20 | `src/myoji/_vendor/` | MyoTorch (formerly MyoVerse), Raul C. Sîmpetru and Vlad Cnejevici | GPL-3.0 (`src/myoji/_vendor/LICENSE-GPL-3.0`) |
| RaulNet V19 (adapted-CNN receiver) | `paper/scripts/vendor/raulnet_v19.py` | MyoTorch | GPL-3.0 |
| One Euro filter (display smoothing only) | `paper/scripts/vendor/one_euro_filter.py` | Nicolas Roussel and Géry Casiez, Inria ([OneEuroFilter](https://github.com/casiez/OneEuroFilter)) | BSD-3-Clause (`paper/scripts/vendor/OneEuroFilter-LICENSE`) |
| Patched CMSIS-DSP header | `paper/embedded/stm32n6_optimized/vendor/cmsis_patch/arm_vec_fft.h` | Arm CMSIS-DSP, modified (see the notice in the file) | Apache-2.0 |
| TBME 2024 CNN (retrained reference receiver) | `paper/scripts/vendor/tbme2024/` | The authors' network from Sîmpetru et al., IEEE TBME 2024 (doi:10.1109/TBME.2024.3432800) | as this repository |

## Used but not included

| what | used by | licence |
|---|---|---|
| [emg2pose](https://github.com/facebookresearch/emg2pose) code, UmeTrack hand model and dataset | `myoji.data`, emg2pose experiments in `paper/` | CC BY-NC-SA 4.0 (UmeTrack hand model: CC BY-NC 4.0) |
| [STM32CubeN6](https://github.com/STMicroelectronics/STM32CubeN6) HAL and CMSIS | `paper/embedded` board builds | BSD-3-Clause / Apache-2.0 |
| Qwen3-0.6B-Base (Hugging Face) | Qwen3 receiver in `paper/scripts` | Apache-2.0 |
| TabPFN 9.0.0 (model v3.5) | TabPFN receiver in `paper/scripts` | Prior Labs licence |
| [BioCodec](https://github.com/klean2050/BioCodec-release) (commit `575bf6e`) | codec comparison in `paper/scripts` | MIT, Kleanthis Avramidis |
