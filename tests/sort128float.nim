import std/[monotimes, random, algorithm, strformat ]
from std/stats import mean
from std/algorithm import sort

import nimsimd/avx2

when defined(gcc) or defined(clang):
  {.localPassc: "-mavx2".}

##[

  This code shows how to sort 128 x float32 without transposing the matrix.

  Project "Fast and Robust"
  paper :: https://drops.dagstuhl.de/storage/00lipics/lipics-vol190-sea2021/LIPIcs.SEA.2021.3/LIPIcs.SEA.2021.3.pdf
  code  :: https://github.com/simd-sorting/fast-and-robust/blob/master/sorting_networks_demos/sort_128_int_no_transpose/sort_128_int_no_transpose.cpp

]##

const
  Mask2301 = MM_SHUFFLE(2,3,0,1)
  Mask0123 = MM_SHUFFLE(0,1,2,3)

# compute 8 modules
#
func COEX( a,b :var M256, c :M256 = a ) =
  a = mm256_min_ps( a, b )
  b = mm256_max_ps( c, b )


# sort 8 columns, each containing 16 x int32 with Green's 60 modules network
#
func sort16IntByColumn( vecsPtr :ptr float32 ) =

  let vecs = cast[ptr UncheckedArray[M256] ]( vecsPtr )

  # step 1
  COEX( vecs[ 0], vecs[ 1]);  COEX( vecs[ 2], vecs[ 3])
  COEX( vecs[ 4], vecs[ 5]);  COEX( vecs[ 6], vecs[ 7])
  COEX( vecs[ 8], vecs[ 9]);  COEX( vecs[10], vecs[11])
  COEX( vecs[12], vecs[13]);  COEX( vecs[14], vecs[15])

  # step 2
  COEX( vecs[ 0], vecs[ 2]);  COEX( vecs[ 1], vecs[ 3])
  COEX( vecs[ 4], vecs[ 6]);  COEX( vecs[ 5], vecs[ 7])
  COEX( vecs[ 8], vecs[10]);  COEX( vecs[ 9], vecs[11])
  COEX( vecs[12], vecs[14]);  COEX( vecs[13], vecs[15])

  # step 3
  COEX( vecs[ 0], vecs[ 4]);  COEX( vecs[ 1], vecs[ 5])
  COEX( vecs[ 2], vecs[ 6]);  COEX( vecs[ 3], vecs[ 7])
  COEX( vecs[ 8], vecs[12]);  COEX( vecs[ 9], vecs[13])
  COEX( vecs[10], vecs[14]);  COEX( vecs[11], vecs[15])

  # step 4
  COEX( vecs[ 0], vecs[ 8]);  COEX( vecs[ 1], vecs[ 9])
  COEX( vecs[ 2], vecs[10]);  COEX( vecs[ 3], vecs[11])
  COEX( vecs[ 4], vecs[12]);  COEX( vecs[ 5], vecs[13])
  COEX( vecs[ 6], vecs[14]);  COEX( vecs[ 7], vecs[15])

  # step 5
  COEX( vecs[ 5], vecs[10]);  COEX( vecs[ 6], vecs[ 9])
  COEX( vecs[ 3], vecs[12]);  COEX( vecs[ 7], vecs[11])
  COEX( vecs[13], vecs[14]);  COEX( vecs[ 4], vecs[ 8])
  COEX( vecs[ 1], vecs[ 2])

  # step 6
  COEX( vecs[ 1], vecs[ 4]);  COEX( vecs[ 7], vecs[13])
  COEX( vecs[ 2], vecs[ 8]);  COEX( vecs[11], vecs[14])

  # step 7
  COEX( vecs[ 2], vecs[ 4]);  COEX( vecs[ 5], vecs[ 6])
  COEX( vecs[ 9], vecs[10]);  COEX( vecs[11], vecs[13])
  COEX( vecs[ 3], vecs[ 8]);  COEX( vecs[ 7], vecs[12])

  # step 8
  COEX( vecs[ 3], vecs[ 5]);  COEX( vecs[ 6], vecs[ 8])
  COEX( vecs[ 7], vecs[ 9]);  COEX( vecs[10], vecs[12])

  # step 9
  COEX( vecs[ 3], vecs[ 4]);  COEX( vecs[ 5], vecs[ 6])
  COEX( vecs[ 7], vecs[ 8]);  COEX( vecs[ 9], vecs[10])
  COEX( vecs[11], vecs[12])

  # step 10
  COEX( vecs[ 6], vecs[ 7]); COEX( vecs[ 8], vecs[ 9])

# merge columns without transposition
#
template MASK( a, b, c, d, e, f, g, h :static int ) :int32 =
  (
    ((h.uint32 < 7).ord shl 7 ) or
    ((g.uint32 < 6).ord shl 6 ) or
    ((f.uint32 < 5).ord shl 5 ) or
    ((e.uint32 < 4).ord shl 4 ) or
    ((d.uint32 < 3).ord shl 3 ) or
    ((c.uint32 < 2).ord shl 2 ) or
    ((b.uint32 < 1).ord shl 1 ) or
    ((a.uint32 < 0).ord )
  ).int32

# the three templates below should be funcs, but
# then performance drops significantly ?
#
template COEX_SHUFFLE*( vec :var M256; a, b, c, d, e, f, g, h: static int ) =
  let
    shuffleVec = vec.mm256_permute_ps MM_SHUFFLE( d, c, b, a )
    minVec     = shuffleVec.mm256_min_ps vec
    maxVec     = shuffleVec.mm256_max_ps vec
  vec = mm256_blend_ps( minVec, maxVec, MASK( a, b, c, d, e, f, g, h ))

template COEX_PERMUTE*( vec :var M256; a, b, c, d, e, f, g, h :static int32 ) =
  let
    permuteMask = mm256_setr_epi32( a, b, c, d, e, f, g, h )
    permuteVec  = vec.mm256_permutevar8x32_ps permuteMask
    minVec      = permuteVec.mm256_min_ps vec
    maxVec      = permuteVec.mm256_max_ps vec
  vec = mm256_blend_ps( minVec, maxVec, MASK(a, b, c, d, e, f, g, h) )

template REVERSE_VEC*( vec :var M256 ) =
  vec = mm256_permutevar8x32_ps(
    vec, mm256_setr_epi32( 7, 6, 5, 4, 3, 2, 1, 0 )
  )


func merge8ColumnsWith16Elements( vecsPtr :ptr float32 ) =
  let vecs = cast[ptr UncheckedArray[M256] ]( vecsPtr )

  vecs[ 8] = mm256_permute_ps( vecs[ 8], Mask2301 );  COEX( vecs[ 7], vecs[ 8] )
  vecs[ 9] = mm256_permute_ps( vecs[ 9], Mask2301 );  COEX( vecs[ 6], vecs[ 9] )
  vecs[10] = mm256_permute_ps( vecs[10], Mask2301 );  COEX( vecs[ 5], vecs[10] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask2301 );  COEX( vecs[ 4], vecs[11] )
  vecs[12] = mm256_permute_ps( vecs[12], Mask2301 );  COEX( vecs[ 3], vecs[12] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask2301 );  COEX( vecs[ 2], vecs[13] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask2301 );  COEX( vecs[ 1], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask2301 );  COEX( vecs[ 0], vecs[15] )
  vecs[ 4] = mm256_permute_ps( vecs[ 4], Mask2301 );  COEX( vecs[ 3], vecs[ 4] )
  vecs[ 5] = mm256_permute_ps( vecs[ 5], Mask2301 );  COEX( vecs[ 2], vecs[ 5] )
  vecs[ 6] = mm256_permute_ps( vecs[ 6], Mask2301 );  COEX( vecs[ 1], vecs[ 6] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask2301 );  COEX( vecs[ 0], vecs[ 7] )
  vecs[12] = mm256_permute_ps( vecs[12], Mask2301 );  COEX( vecs[11], vecs[12] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask2301 );  COEX( vecs[10], vecs[13] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask2301 );  COEX( vecs[ 9], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask2301 );  COEX( vecs[ 8], vecs[15] )
  vecs[ 2] = mm256_permute_ps( vecs[ 2], Mask2301 );  COEX( vecs[ 1], vecs[ 2] )
  vecs[ 3] = mm256_permute_ps( vecs[ 3], Mask2301 );  COEX( vecs[ 0], vecs[ 3] )
  vecs[ 6] = mm256_permute_ps( vecs[ 6], Mask2301 );  COEX( vecs[ 5], vecs[ 6] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask2301 );  COEX( vecs[ 4], vecs[ 7] )
  vecs[10] = mm256_permute_ps( vecs[10], Mask2301 );  COEX( vecs[ 9], vecs[10] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask2301 );  COEX( vecs[ 8], vecs[11] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask2301 );  COEX( vecs[13], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask2301 );  COEX( vecs[12], vecs[15] )
  vecs[ 1] = mm256_permute_ps( vecs[ 1], Mask2301 );  COEX( vecs[ 0], vecs[ 1] )
  vecs[ 3] = mm256_permute_ps( vecs[ 3], Mask2301 );  COEX( vecs[ 2], vecs[ 3] )
  vecs[ 5] = mm256_permute_ps( vecs[ 5], Mask2301 );  COEX( vecs[ 4], vecs[ 5] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask2301 );  COEX( vecs[ 6], vecs[ 7] )
  vecs[ 9] = mm256_permute_ps( vecs[ 9], Mask2301 );  COEX( vecs[ 8], vecs[ 9] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask2301 );  COEX( vecs[10], vecs[11] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask2301 );  COEX( vecs[12], vecs[13] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask2301 );  COEX( vecs[14], vecs[15] )
  COEX_SHUFFLE( vecs[ 0], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[ 1], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 2], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[ 3], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 4], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[ 5], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 6], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[ 7], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 8], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[ 9], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[10], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[11], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[12], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[13], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[14], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_SHUFFLE( vecs[15], 1, 0, 3, 2, 5, 4, 7, 6 )
  vecs[ 8] = mm256_permute_ps( vecs[ 8], Mask0123 );  COEX( vecs[ 7], vecs[ 8] )
  vecs[ 9] = mm256_permute_ps( vecs[ 9], Mask0123 );  COEX( vecs[ 6], vecs[ 9] )
  vecs[10] = mm256_permute_ps( vecs[10], Mask0123 );  COEX( vecs[ 5], vecs[10] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask0123 );  COEX( vecs[ 4], vecs[11] )
  vecs[12] = mm256_permute_ps( vecs[12], Mask0123 );  COEX( vecs[ 3], vecs[12] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask0123 );  COEX( vecs[ 2], vecs[13] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask0123 );  COEX( vecs[ 1], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask0123 );  COEX( vecs[ 0], vecs[15] )
  vecs[ 4] = mm256_permute_ps( vecs[ 4], Mask0123 );  COEX( vecs[ 3], vecs[ 4] )
  vecs[ 5] = mm256_permute_ps( vecs[ 5], Mask0123 );  COEX( vecs[ 2], vecs[ 5] )
  vecs[ 6] = mm256_permute_ps( vecs[ 6], Mask0123 );  COEX( vecs[ 1], vecs[ 6] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask0123 );  COEX( vecs[ 0], vecs[ 7] )
  vecs[12] = mm256_permute_ps( vecs[12], Mask0123 );  COEX( vecs[11], vecs[12] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask0123 );  COEX( vecs[10], vecs[13] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask0123 );  COEX( vecs[ 9], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask0123 );  COEX( vecs[ 8], vecs[15] )
  vecs[ 2] = mm256_permute_ps( vecs[ 2], Mask0123 );  COEX( vecs[ 1], vecs[ 2] )
  vecs[ 3] = mm256_permute_ps( vecs[ 3], Mask0123 );  COEX( vecs[ 0], vecs[ 3] )
  vecs[ 6] = mm256_permute_ps( vecs[ 6], Mask0123 );  COEX( vecs[ 5], vecs[ 6] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask0123 );  COEX( vecs[ 4], vecs[ 7] )
  vecs[10] = mm256_permute_ps( vecs[10], Mask0123 );  COEX( vecs[ 9], vecs[10] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask0123 );  COEX( vecs[ 8], vecs[11] )
  vecs[14] = mm256_permute_ps( vecs[14], Mask0123 );  COEX( vecs[13], vecs[14] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask0123 );  COEX( vecs[12], vecs[15] )
  vecs[ 1] = mm256_permute_ps( vecs[ 1], Mask0123 );  COEX( vecs[ 0], vecs[ 1] )
  vecs[ 3] = mm256_permute_ps( vecs[ 3], Mask0123 );  COEX( vecs[ 2], vecs[ 3] )
  vecs[ 5] = mm256_permute_ps( vecs[ 5], Mask0123 );  COEX( vecs[ 4], vecs[ 5] )
  vecs[ 7] = mm256_permute_ps( vecs[ 7], Mask0123 );  COEX( vecs[ 6], vecs[ 7] )
  vecs[ 9] = mm256_permute_ps( vecs[ 9], Mask0123 );  COEX( vecs[ 8], vecs[ 9] )
  vecs[11] = mm256_permute_ps( vecs[11], Mask0123 );  COEX( vecs[10], vecs[11] )
  vecs[13] = mm256_permute_ps( vecs[13], Mask0123 );  COEX( vecs[12], vecs[13] )
  vecs[15] = mm256_permute_ps( vecs[15], Mask0123 );  COEX( vecs[14], vecs[15] )
  COEX_SHUFFLE( vecs[ 0], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 0], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 1], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 1], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 2], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 2], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 3], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 3], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 4], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 4], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 5], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 5], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 6], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 6], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 7], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 7], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 8], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 8], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[ 9], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[ 9], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[10], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[10], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[11], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[11], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[12], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[12], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[13], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[13], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[14], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[14], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_SHUFFLE( vecs[15], 3, 2, 1, 0, 7, 6, 5, 4 );  COEX_SHUFFLE( vecs[15], 1, 0, 3, 2, 5, 4, 7, 6 )
  REVERSE_VEC(  vecs[ 8] ); COEX( vecs[ 7], vecs[ 8] );   REVERSE_VEC(  vecs[ 9] );  COEX( vecs[ 6], vecs[ 9] )
  REVERSE_VEC(  vecs[10] ); COEX( vecs[ 5], vecs[10] );   REVERSE_VEC(  vecs[11] );  COEX( vecs[ 4], vecs[11] )
  REVERSE_VEC(  vecs[12] ); COEX( vecs[ 3], vecs[12] );   REVERSE_VEC(  vecs[13] );  COEX( vecs[ 2], vecs[13] )
  REVERSE_VEC(  vecs[14] ); COEX( vecs[ 1], vecs[14] );   REVERSE_VEC(  vecs[15] );  COEX( vecs[ 0], vecs[15] )
  REVERSE_VEC(  vecs[ 4] ); COEX( vecs[ 3], vecs[ 4] );   REVERSE_VEC(  vecs[ 5] );  COEX( vecs[ 2], vecs[ 5] )
  REVERSE_VEC(  vecs[ 6] ); COEX( vecs[ 1], vecs[ 6] );   REVERSE_VEC(  vecs[ 7] );  COEX( vecs[ 0], vecs[ 7] )
  REVERSE_VEC(  vecs[12] ); COEX( vecs[11], vecs[12] );   REVERSE_VEC(  vecs[13] );  COEX( vecs[10], vecs[13] )
  REVERSE_VEC(  vecs[14] ); COEX( vecs[ 9], vecs[14] );   REVERSE_VEC(  vecs[15] );  COEX( vecs[ 8], vecs[15] )
  REVERSE_VEC(  vecs[ 2] ); COEX( vecs[ 1], vecs[ 2] );   REVERSE_VEC(  vecs[ 3] );  COEX( vecs[ 0], vecs[ 3] )
  REVERSE_VEC(  vecs[ 6] ); COEX( vecs[ 5], vecs[ 6] );   REVERSE_VEC(  vecs[ 7] );  COEX( vecs[ 4], vecs[ 7] )
  REVERSE_VEC(  vecs[10] ); COEX( vecs[ 9], vecs[10] );   REVERSE_VEC(  vecs[11] );  COEX( vecs[ 8], vecs[11] )
  REVERSE_VEC(  vecs[14] ); COEX( vecs[13], vecs[14] );   REVERSE_VEC(  vecs[15] );  COEX( vecs[12], vecs[15] )
  REVERSE_VEC(  vecs[ 1] ); COEX( vecs[ 0], vecs[ 1] );   REVERSE_VEC(  vecs[ 3] );  COEX( vecs[ 2], vecs[ 3] )
  REVERSE_VEC(  vecs[ 5] ); COEX( vecs[ 4], vecs[ 5] );   REVERSE_VEC(  vecs[ 7] );  COEX( vecs[ 6], vecs[ 7] )
  REVERSE_VEC(  vecs[ 9] ); COEX( vecs[ 8], vecs[ 9] );   REVERSE_VEC(  vecs[11] );  COEX( vecs[10], vecs[11] )
  REVERSE_VEC(  vecs[13] ); COEX( vecs[12], vecs[13] );   REVERSE_VEC(  vecs[15] );  COEX( vecs[14], vecs[15] )
  COEX_PERMUTE( vecs[ 0], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[ 0], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[ 0], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[ 1], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[ 1], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[ 1], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[ 2], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[ 2], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[ 2], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[ 3], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[ 3], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[ 3], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[ 4], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[ 4], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[ 4], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[ 5], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[ 5], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[ 5], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[ 6], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[ 6], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[ 6], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[ 7], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[ 7], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[ 7], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[ 8], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[ 8], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[ 8], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[ 9], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[ 9], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[ 9], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[10], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[10], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[10], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[11], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[11], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[11], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[12], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[12], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[12], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[13], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[13], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[13], 1, 0, 3, 2, 5, 4, 7, 6 )
  COEX_PERMUTE( vecs[14], 7, 6, 5, 4, 3, 2, 1, 0 );  COEX_SHUFFLE( vecs[14], 2, 3, 0, 1, 6, 7, 4, 5 )
  COEX_SHUFFLE( vecs[14], 1, 0, 3, 2, 5, 4, 7, 6 );  COEX_PERMUTE( vecs[15], 7, 6, 5, 4, 3, 2, 1, 0 )
  COEX_SHUFFLE( vecs[15], 2, 3, 0, 1, 6, 7, 4, 5 );  COEX_SHUFFLE( vecs[15], 1, 0, 3, 2, 5, 4, 7, 6 )
  

# sort 16 vectors (128 x float32)
#
func sort128IntWithoutTransposition( vecsPtr :ptr float32 ) =
  sort16IntByColumn vecsPtr
  merge8ColumnsWith16Elements vecsPtr

func sort128*( vecsPtr :ptr float32 ) =
  sort128IntWithoutTransposition vecsPtr

# ======================= EO - Quicksort ==================================

randomize()
proc ns*() :int64 = monotimes.getMonoTime().ticks

type Data = object
  start  {.align(32).} :array[128, float32]

proc mkResults( algo :string, runs :var seq[int] ) :float =
  runs.sort()
  let (best, worst) = ( runs[0], runs[^1] )
  runs = runs[ 1 .. ^2 ]
  echo runs.len, "x", algo, " (", best, ")-", runs, "-(", worst, ")"
  result = stats.mean runs

proc testVQS( data :var Data, samples :int = 12 ) :float =
  var runs :seq[int]
  for r in 0 ..< samples :
    let t0 = ns()
    sort128 data.start[0].addr
    runs.add (ns() - t0).int
    shuffle data.start

  return "VQS".mkResults runs

proc testStd( data :var Data, samples :int = 12 ) :float =
  var runs :seq[int]
  for r in 0 ..< samples:
    let t0 = ns()
    data.start.sort()
    runs.add (ns() - t0).int
    shuffle data.start

  return "Std".mkResults runs

# ======================= main ==================================

when isMainModule:

  var data = Data()
  for i in 0 ..< 128:
    data.start[i] = rand(10_000).float32

  # take 12-samples
  # drop the best- and the worst-sample
  # return the mean-of-remaining 10-samples
  #
  let
    vqsMean = testVQS data
    stdMean = testStd data
    theX    = stdMean / vqsMean

    p1 = fmt"std/sort {stdMean:.2f}"
    p2 = fmt" | ~{theX:.1f}X | "
    p3 = fmt"{vqsMean:.2f} vectorized Quicksort"

  echo p1,p2,p3