
##[
This implementation of `xoroshiro128+ <https://en.wikipedia.org/wiki/Xoroshiro128%2B>`_ was done by
Mark Blacher et al.
..   [Blacher2021] Blacher, Giesen, und Kühne, „Fast and Robust Vectorized In-Place Sorting of Primitive Types“ doi 10.4230/LIPIcs.SEA.2021.3
 ]##

import nimsimd/avx2
import common_avx2

when defined(gcc) or defined(clang):
  {.localPassc: "-mavx2".}


# transform random numbers to the range between 0 and bound - 1
#
func rndEpu32*( rndVec, boundVec :M256i ) :M256i =
  let
    evenVec = rndVec.mul( boundVec, epu32 ).shiftLeft( 32'i32, epi64 )
    oddVec  = rndVec.shiftLeft(32'i32, epi64 ).mul( boundVec, epu32 )

  result = oddVec.blend( evenVec, 0b01_01_01_01, epi32 )


# vectorized random-number-generator 'xoroshiro128+'
#
func VROTL*( x :M256i, k :int32 ) :M256i =
  #
  # rotate each uint64 value in vector
  #
  result = x.shiftLeft(k) or x.shiftLeft( 64'i32 - k )


func vNext*( seedA,seedB :var M256i ) :M256i =

  seedB  = seedA or seedB       # modify vectors seedA and seedB
  seedA  = ( VROTL( seedA, 24'i32 ) xor seedB) xor seedB.shiftLeft( 16'i32 )
  seedB  = VROTL( seedB, 37'i32)
  result = seedA.add seedB  # return a random vector