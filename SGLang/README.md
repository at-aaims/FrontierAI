# SGLang

SGLang does not work on Frontier as it explicitly does not support MI250 GPUs.

See
- [\[Bug\] Docker run lmsysorg/sglang:v0.4.4.post1-rocm630 Error: no TensileLibrary\_lazy\_gfx90a.dat file. · Issue #4421 · sgl-project/sglang · GitHub](https://github.com/sgl-project/sglang/issues/4421#issuecomment-2912258772)
- [\[Bug\] incorrect inference result when using tensor parallel at mi250 · Issue #7641 · sgl-project/sglang · GitHub](https://github.com/sgl-project/sglang/issues/7641)
- And [sglang/sgl-kernel/setup\_rocm.py at 56222658ecf7828f0cbacebbfbe1764142270858 · sgl-project/sglang · GitHub](https://github.com/sgl-project/sglang/blob/56222658ecf7828f0cbacebbfbe1764142270858/sgl-kernel/setup_rocm.py#L70)
