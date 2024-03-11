# Vectorized Quicksort 'VQsort'
This is a pure [Nim](https://nim-lang.org)-version of *VQsort* (AVX2-version) based on the 2021 [article/PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol190-sea2021/LIPIcs.SEA.2021.3/LIPIcs.SEA.2021.3.pdf) by Blacher et al.
This implementation is for educational purposes. It can sort 32-bit Integers (and soon Floats).

Find the genuine C++ implementation of their project **"Fast and Robust"** at [github-repo](https://github.com/simd-sorting/fast-and-robust).

### Note

The google-brain-group has developed a advanced version of this algorithm. It's based on googles Highway-library and thus supports Intel/ARM/RISC-V and all mainline compilers. Find it at [github/google/highway/contrib/sort](https://github.com/google/highway/tree/master/hwy/contrib/sort).
Their advanced algorithm adapts to the SIMD-capabillities of the targeted platform. Furthermore it can sort 8/16/32/64/128-Bit Integers and Floats. For a detailed description consult this preprint [Vectorized and performance-portable Quicksort](http://arxiv.org/abs/2205.05982) [ retrieved May.2022].
