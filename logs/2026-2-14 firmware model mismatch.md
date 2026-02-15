# Troubleshooting Log - 2026-02-14

## Goal

Pass a captured photo from `app.py` into the Qwen2-VL-2B HEF model on the
Hailo-10H and get text output.

---

## Issue 1: VLM stub not wired to Hailo GenAI API

**Symptom:** `vlm.py` had placeholder code that always returned
`"[Qwen2-VL on Hailo-10H: integration pending.]"` instead of running inference.

**Root cause:** The `HailoQwen2VL` class was a stub. The Hailo GenAI Python API
(`hailo_platform.genai.VLM`) was not imported or called.

**Fix:** Rewrote `vlm.py` to:
- Import `VDevice` and `VLM` from `hailo_platform` / `hailo_platform.genai`
- Initialize the device and load the HEF in `__init__`
- Query `input_frame_shape()` / `input_frame_format_type()` for preprocessing
- Implement `describe_image()`: resize frame with `cv2.resize`, build a
  structured prompt, call `vlm.generate_all()`, and return the text
- Default HEF path changed from `~/Downloads/` to `models/Qwen2-VL-2B-Instruct.hef`
  relative to project directory

**Reference:** Official Hailo VLM tutorial at
`/usr/lib/python3/dist-packages/hailo_tutorials/notebooks/HRT_4_VLM_Tutorial.ipynb`

---

## Issue 2: HAILO_INVALID_OPERATION(6) - Failed to create VLM

**Symptom:** When `VLM(vdevice, hef_path)` was called, the Hailo-10H device
returned `HAILO_INVALID_OPERATION` (status 6). Error originated from
`serializer.cpp:996` in `deserialize_reply`, meaning the device firmware itself
was rejecting the VLM creation request.

```
[HailoRT] [error] [serializer.cpp:996] [deserialize_reply] CHECK_SUCCESS failed
  with status=HAILO_INVALID_OPERATION(6) - Failed to create VLM
```

**Debugging steps:**
1. Verified the Hailo-10H device was present and healthy
   (`hailortcli fw-control identify` returned firmware 5.1.1, HAILO10H).
2. Verified no other processes were using the device.
3. Validated the HEF with `hailortcli parse-hef` -- compatible with HAILO10H,
   3 network groups (encoder, prefill, tbt).
4. Tried `optimize_memory_on_device=True` -- same error.
5. Tried different VDevice scheduling algorithms (NONE, ROUND_ROBIN) -- same error,
   and NONE was explicitly not supported for GenAI.
6. Successfully loaded a different GenAI model (Llama 3.2 3B LLM) on the same
   device, proving GenAI support works at the firmware level.
7. Checked `hailort.log` -- confirmed the error came from the device firmware
   reply, not host-side processing.
8. Compared HailoRT v5.1.1 to v5.2.0 on GitHub and found breaking changes in
   `genai_scheme.proto` (VLM_Create_Request field types changed) and `hef.proto`
   (3 new lines).

**Root cause:** The Qwen2-VL-2B HEF file was downloaded from hailo.ai on
January 31, 2026 -- after HailoRT v5.2.0 was released on January 8, 2026.
Hailo's website served an HEF compiled with the v5.2.0 Dataflow Compiler.
The v5.1.1 firmware on the device could not process the newer HEF format for
VLM creation, returning HAILO_INVALID_OPERATION.

**Fix:** Upgrade the entire HailoRT stack to v5.2.0:
1. Downloaded `hailort-pcie-driver_5.2.0_all.deb` and `hailort_5.2.0_arm64.deb`
   from the Hailo Developer Zone (requires free account).
2. Removed old RPi-specific packages:
   `sudo dpkg -r hailo-h10-all python3-h10-hailort`
   `sudo dpkg -r --force-depends h10-hailort h10-hailort-pcie-driver`
3. Installed new packages:
   `sudo dpkg -i hailort-pcie-driver_5.2.0_all.deb`
   `sudo dpkg -i hailort_5.2.0_arm64.deb`
4. The generic Ubuntu deb did not include Python bindings. Built them from
   source (v5.2.0 tag from github.com/hailo-ai/hailort):
   ```
   cd hailort/libhailort/bindings/python/src
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
     -DLIBHAILORT_PATH=/usr/lib/libhailort.so \
     -DHAILORT_INCLUDE_DIR=/usr/include/hailo \
     -DPYBIND11_PYTHON_VERSION=3.13
   cmake --build build --config Release -j$(nproc)
   ```
   Copied the built `_pyhailort.cpython-313-aarch64-linux-gnu.so` and Python
   wrapper files to `/usr/lib/python3/dist-packages/hailo_platform/`.
5. Reboot required to load new 5.2.0 firmware onto the Hailo-10H device.

---

## Resolution

After rebooting, the Hailo-10H device loaded firmware 5.2.0 successfully.
VLM creation and inference now work end-to-end:

- **Firmware:** 5.2.0 (confirmed via `hailortcli fw-control identify`)
- **VLM model:** Qwen2-VL-2B-Instruct.hef loads in ~12 seconds
- **Frame input:** 336x336x3 uint8 (RGB)
- **Max context:** 2048 tokens
- **Inference time:** ~12.6 seconds for 100 tokens on a test image
- **Output:** coherent, descriptive English text

The full pipeline (capture frame -> resize -> VLM inference -> text output)
is now operational through `app.py` and `vlm.py`.
