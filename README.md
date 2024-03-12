# Vectorized Quicksort 'VQsort'
This is a pure [Nim](https://nim-lang.org)-version of *VQsort* (AVX2-version) based on the article *Fast and Robust Vectorized In-Place Sorting of Primitive Types* from 2021 [PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol190-sea2021/LIPIcs.SEA.2021.3/LIPIcs.SEA.2021.3.pdf) by Blacher et al.
It combines sorting-networks (4/8/16) with bitonic merge procedures, a tiny pseudo-random generator (xoroshiro128+) on registers and a twofold pivot-selection strategy. The authors claim to dethrone Intels de-facto best performing sort algorithm. I can confirm its performance is amazing.
This implementation is for educational purposes. It can sort 32-bit Integers (and soon floats).
So this marks the return of Quicksort to the top of the food-chain of sorting algorithms.
Find the genuine C++ implementation of their project **"Fast and Robust"** at [github-repo](https://github.com/simd-sorting/fast-and-robust).

### Note

According to this [blog-post](https://opensource.googleblog.com/2022/06/Vectorized%20and%20performance%20portable%20Quicksort.html) of June.2022 by Jan Wassenberg, it seems  Googles Brain-group has developed a advanced version of VQSort. It's based on their Highway-library and can support Intel-/ARM-/RISC-V-SIMD including all mainline compilers. If you are after a production-ready algorithm you can find it at [github/google/highway/contrib/sort](https://github.com/google/highway/tree/master/hwy/contrib/sort).
Their advanced version adapts to the SIMD-capabillities of the targeted platform - including AVX-512 - and does multithreading. Their paper says this gives another 1.5-2.8-X. Furthermore it can sort 8/16/32/64/128-Bit Integers and Floats. For a detailed description consult this preprint [Vectorized and performance-portable Quicksort](http://arxiv.org/abs/2205.05982) [ retrieved May.2022].
