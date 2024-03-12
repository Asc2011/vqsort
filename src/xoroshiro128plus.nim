
##[
This implementation of `xoroshiro128+ <https://en.wikipedia.org/wiki/Xoroshiro128%2B>`_ was done by
Mark Blacher et al.
..   [Blacher2021] Blacher, Giesen, und Kühne, „Fast and Robust Vectorized In-Place Sorting of Primitive Types“ doi 10.4230/LIPIcs.SEA.2021.3
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