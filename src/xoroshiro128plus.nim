
##[
This implementation of `xoroshiro128+ <https://en.wikipedia.org/wiki/Xoroshiro128%2B>`_ was done by
Mark Blacher et al. It was published 2021 in their paper *'Fast and Robust Vectorized In-Place Sorting of Primitive Types'* [Blacher2021]_.
It provides pseudo-randomness for their vectorized Quicksort Algorithm by the name **VQSort**.
It is part of the `"Fast and Robust"-repository <https://github.com/simd-sorting/fast-and-robust>`_.
Following this `blog-post <https://opensource.googleblog.com/2022/06/Vectorized%20and%20performance%20portable%20Quicksort.html>`_
the further development of the *VQSort*-algorithm goes on at Googles Brain group under the supervision
of Jan Wassenberg. The newer codebase uses Googles' Highway-library and remains open
as a contribution to sorting at `github/google/highway/contrib/sort <https://github.com/google/highway/tree/master/hwy/contrib/sort>`_.
Within this preprint [Blacher20220513]_ the authors describe their general progress on *VQSort* and the changes
to pseudo random-generation in detail. Note this code was adapted from the earlier publication *'Fast and Robust'* in 2021.

..   [Blacher2021] Blacher, Giesen, und Kühne, „Fast and Robust Vectorized In-Place Sorting of Primitive Types“ doi 10.4230/LIPIcs.SEA.2021.3
..   [Blacher20220513] „Vectorized and performance-portable Quicksort“. retrieved 10.march.2024

 ]##

import nimsimd/avx2

when defined(gcc) or defined(clang):
  {.localPassc: "-mavx2".}


# transform random numbers to the range between 0 and bound - 1
#
func rndEpu32*( rndVec, boundVec :M256i ) :M256i =
  let
    evenVec = mm256_srli_epi64(  mm256_mul_epu32(  rndVec, boundVec ), 32'i32 )
    oddVec  = mm256_mul_epu32(   mm256_srli_epi64( rndVec, 32'i32), boundVec  )

  result = oddVec.mm256_blend_epi32( evenVec, 0b01_01_01_01 )

# vectorized random-number-generator 'xoroshiro128+'
#
func VROTL*( x :M256i, k :int32 ) :M256i =
  #
  # rotate each uint64 value in vector
  #
  mm256_or_si256(
    x.mm256_slli_epi64 k ,
    x.mm256_srli_epi64 64'i32 - k
  )


func vNext*( seedA,seedB :var M256i ) :M256i =

  seedB = seedA.mm256_xor_si256 seedB       # modify vectors seedA and seedB
  seedA = mm256_xor_si256(
    mm256_xor_si256( seedA.VROTL( 24'i32 ), seedB ),
    seedB.mm256_slli_epi64 16'i32
  )
  seedB  = seedB.VROTL 37'i32
  result = seedA.mm256_add_epi64 seedB  # return a random vector