import nimsimd/avx2

export avx2

when defined(gcc) or defined(clang):
  {.localPassc: "-mavx2".}

import xoroshiro128plus

# a (2-Kb) static mask-lookup-table
#
include maskLUTarr

const
  Mask2301 = MM_SHUFFLE(2, 3, 0, 1)
  Mask0123 = MM_SHUFFLE(0, 1, 2, 3)
  init07MaskArr  :array[8, int32] = [ 0, 1, 2, 3, 4, 5, 6, 7 ]
  revertMaskArr  :array[8, int32] = [ 7, 6, 5, 4, 3, 2, 1, 0 ]
  restoreMaskArr :array[8, int32] = [ 0, 4, 1, 5, 6, 2, 7, 3 ]

func load(  arrPt :pointer )    :M256i   = mm256_loadu_si256   arrPt
func store( arrPt :pointer, vec :M256i ) = mm256_storeu_si256( arrPt, vec )

func average( a,b :int32 ) :int32 = (a and b) + ((a xor b) shr 1)

func shuffle( vecA,vecB :M256i; imm8 :static int32 ) :M256i =
  mm256_castps_si256(
    mm256_shuffle_ps(
      vecA.mm256_castsi256_ps, vecB.mm256_castsi256_ps, imm8
    )
  )

func reverse( vec :var M256i ) =
  let revertMaskV = load revertMaskArr.addr
  vec = vec.mm256_permutevar8x32_epi32 revertMaskV

func calcMin( vec :var M256i ) :int32 = # minimum of 8 x int32
  let revertMaskV = load revertMaskArr.addr
  vec = vec.mm256_min_epi32( vec.mm256_permutevar8x32_epi32 revertMaskV )
  vec = vec.mm256_min_epi32( vec.mm256_shuffle_epi32 0b10110001 )
  vec = vec.mm256_min_epi32( vec.mm256_shuffle_epi32 0b01001110 )
  result = vec.mm256_extract_epi32 0'i32

func calcMax( vec :var M256i ) :int32 = # maximum of 8 x int32
  let revertMaskV = load revertMaskArr.addr
  vec = vec.mm256_max_epi32( vec.mm256_permutevar8x32_epi32 revertMaskV )
  vec = vec.mm256_max_epi32( vec.mm256_shuffle_epi32 0b10110001 )
  vec = vec.mm256_max_epi32( vec.mm256_shuffle_epi32 0b01001110 )
  result = vec.mm256_extract_epi32 0'i32


template bitMask( a, b, c, d, e, f, g, h :static int ) :int32 =
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

func COEX( vecA,vecB :var M256i; vecC :M256i = vecA ) =
  vecA = vecA.mm256_min_epi32 vecB
  vecB = vecC.mm256_max_epi32 vecB

func COEXshuffle*( vec :var M256i; a, b, c, d, e, f, g, h :static int ) =
  let
    shuffled    = vec.mm256_shuffle_epi32 MM_SHUFFLE(d, c, b, a)
    minVec      = shuffled.mm256_min_epi32 vec
    maxVec      = shuffled.mm256_max_epi32 vec
  vec = mm256_blend_epi32( minVec, maxVec, bitMask(a, b, c, d, e, f, g, h))

func COEXpermute*( vec :var M256i; a, b, c, d, e, f, g, h :static int32 ) =
  let
    permuteMaskV = mm256_setr_epi32( a, b, c, d, e, f, g, h )
    permuted     = vec.mm256_permutevar8x32_epi32 permuteMaskV
    minVec       = permuted.mm256_min_epi32 vec
    maxVec       = permuted.mm256_max_epi32 vec
  vec = mm256_blend_epi32( minVec, maxVec, bitMask(a, b, c, d, e, f, g, h) )


# sorting-network for 8 x int32 with compare-exchange macros
# ( used for pivot selection in median of the medians ).
#
proc sort8( vec :var M256i ) =
  COEXshuffle( vec, 1, 0, 3, 2, 5, 4, 7, 6 )
  COEXshuffle( vec, 2, 3, 0, 1, 6, 7, 4, 5 )
  COEXshuffle( vec, 0, 2, 1, 3, 4, 6, 5, 7 )
  COEXpermute( vec, 7, 6, 5, 4, 3, 2, 1, 0 )
  COEXshuffle( vec, 2, 3, 0, 1, 6, 7, 4, 5 )
  COEXshuffle( vec, 1, 0, 3, 2, 5, 4, 7, 6 )


# optimized sorting-network for two vectors, that is 16 x int32.
#
proc sort16( v1,v2 :var M256i ) =

  COEX(v1, v2)                                          # step 1

  v2 = v2.mm256_shuffle_epi32 Mask2301                  # step 2
  COEX(v1, v2)

  var tmp = v1                                          # step 3
  v1 = shuffle(  v1, v2, 0b10001000 )
  v2 = shuffle( tmp, v2, 0b11011101 )
  COEX(v1, v2)

  v2 = v2.mm256_shuffle_epi32 Mask0123                 # step  4
  COEX(v1, v2)

  tmp = v1                                              # step  5
  v1 = shuffle(  v1, v2, 0b01000100 )
  v2 = shuffle( tmp, v2, 0b11101110 )
  COEX(v1, v2)

  tmp = v1                                              # step 6
  v1 = shuffle(  v1, v2, 0b11011000 )
  v2 = shuffle( tmp, v2, 0b10001101 )
  COEX(v1, v2)

  v2 = v2.mm256_permutevar8x32_epi32( load revertMaskArr.addr )
  COEX(v1, v2)                                          # step 7

  tmp = v1                                              # step 8
  v1 = shuffle(  v1, v2, 0b11011000 )
  v2 = shuffle( tmp, v2, 0b10001101 )
  COEX(v1, v2)

  tmp = v1                                              # step 9
  v1 = shuffle(  v1, v2, 0b11011000 )
  v2 = shuffle( tmp, v2, 0b10001101 )
  COEX(v1, v2)

  # permute to make it easier to restore order
  #
  let restoreMaskV = load restoreMaskArr.addr
  v1 = v1.mm256_permutevar8x32_epi32 restoreMaskV
  v2 = v2.mm256_permutevar8x32_epi32 restoreMaskV

  tmp = v1                                              # step 10
  v1 = shuffle(  v1, v2, 0b10001000 )
  v2 = shuffle( tmp, v2, 0b11011101 )
  COEX(v1, v2)

  # restore order
  #
  let
    b2 = v2.mm256_shuffle_epi32 0b10_11_00_01
    b1 = v1.mm256_shuffle_epi32 0b10_11_00_01
  v1   = v1.mm256_blend_epi32( b2, 0b10_10_10_10 )
  v2   = b1.mm256_blend_epi32( v2, 0b10_10_10_10 )


# partition a single vector, return how many values are greater than pivot,
# update smallest- and largest-values in smallestVec and biggestVec respectively.
#
proc partitionVec( currVec, pivotVec, smallestVec, biggestVec :var M256i ) :int32 =
  #
  # which elements are larger than the pivot ?
  #
  let cmpVec = currVec.mm256_cmpgt_epi32 pivotVec
  #
  # update the smallest and largest values of the array
  #
  smallestVec = currVec.mm256_min_epi32 smallestVec
  biggestVec  = currVec.mm256_max_epi32 biggestVec
  #
  # extract the most significant bit from each integer of the vector
  #
  let mm = mm256_movemask_ps( mm256_castsi256_ps cmpVec )
  #
  # how many ones ? each 1 stands for an element greater than pivot
  #
  let countGTpivot = mm_popcnt_u32( mm.uint32 ).int32
  #
  # permute elements larger than pivot to the right, and,
  # smaller than or equal to the pivot, to the left.
  #
  currVec = currVec.mm256_permutevar8x32_epi32(
   load permutationMasks[mm*8].addr
  )
  # return how many elements are greater than pivot
  #
  result = countGTpivot


include partitionVectorized8
include partitionVectorized64
include bitonicMerge
include merge8ColsWith16Elems


# sort 8-columns, each containing 16 x int32 with Green's 60-modules network.
#
func sort16Int32Vertical( bufPtr :pointer ) =

  let vecs = cast[ptr UncheckedArray[M256i] ]( bufPtr )

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


proc sortInt32Network( arrPtr :pointer, bufPtr :pointer, n :int32 ) =

  if n < 2: return

  let
    arr    = cast[ptr UncheckedArray[int32]]( arrPtr )
    buffer = cast[ptr UncheckedArray[M256i]]( bufPtr )

    remainder = if (let rem = n mod 8; rem) > 0 :
      rem
    else:
      8
    idxMaxPad = n - remainder

    init07MaskV = load init07MaskArr.addr
    maskVec     = mm256_add_epi32( mm256_set1_epi32( -remainder ), init07MaskV )
    maxPadVec   = mm256_blendv_epi8(
      mm256_set1_epi32 int32.high ,
      mm256_maskload_epi32( arr[idxMaxPad].addr, maskVec ),
      maskVec
    )

  for i in 0 ..< (idxMaxPad div 8) :
    buffer[i] = load arr[i*8].addr
    assert i < 66, "sortInt32Network:: overflow in i-" & $i

  buffer[idxMaxPad div 8  ]   = maxPadVec
  buffer[idxMaxPad div 8 + 1] = mm256_set1_epi32 int32.high

  let N = ((idxMaxPad mod 16 == 0 ).int32 * 8 + idxMaxPad + 8) div 8

  for j in countup( 0, (N-(N mod 16)).pred, 16 ) :
    sort16Int32Vertical buffer[j].addr
    merge8ColumnsWith16Elements buffer[j].addr

  for i in countup( N-N mod 16, N.pred, 2 ) : sort16( buffer[i], buffer[i.succ] )

  let modN16 = N mod 16
  bitonicMerge16(  buffer[N - modN16].addr, modN16 )
  bitonicMerge128( buffer, N )

  for i in 0 ..< (idxMaxPad div 8) : store( arr[i*8].addr, buffer[i] )

  mm256_maskstore_epi32( arr[idxMaxPad].addr, maskVec, buffer[idxMaxPad div 8] )


func getPivot( arr :ptr UncheckedArray[int32], left, right :int32 ) :int32 =

  #debugEcho fmt"getPivot:: left-{left} right-{right}"

  let
    boundVec = mm256_set1_epi32 right - left.succ
    leftVec  = mm256_set1_epi32 left

  # seeds for the vectorized random-number-generator.
  #
  var
    seedVa = mm256_setr_epi64x(  8265987198341093849, 3762817312854612374,
                                 1324281658759788278, 6214952190349879213 )
    seedVb = mm256_setr_epi64x(  2874178529384792648, 1257248936691237653,
                                 7874578921548791257, 1998265912745817298 )

  seedVa = seedVa.mm256_add_epi64( mm256_set1_epi64x  left.int64 )
  seedVb = seedVb.mm256_sub_epi64( mm256_set1_epi64x right.int64 )

  var v :array[9, M256i]
  var resVec :M256i
  for i in 0 ..< 9 :                        # fill 9 vectors with random numbers
    resVec = seedVa.vNext seedVb            # vector with 4 random epu64/uint64
    resVec = rndEpu32( resVec, boundVec )   # random numbers between 0 and boundVec - 1
    resVec = resVec.mm256_add_epi32 leftVec # indices for arr
    v[i]   = mm256_i32gather_epi32( arr[0].addr, resVec, 4'i32 )


  # median network for 9-elements
  #
  COEX( v[0], v[1] );  COEX( v[2], v[3] )   # step 1
  COEX( v[4], v[5] );  COEX( v[6], v[7] )
  COEX( v[0], v[2] );  COEX( v[1], v[3] )   # step 2
  COEX( v[4], v[6] );  COEX( v[5], v[7] )
  COEX( v[0], v[4] );  COEX( v[1], v[2] )   # step 3
  COEX( v[5], v[6] );  COEX( v[3], v[7] )
  COEX( v[1], v[5] );  COEX( v[2], v[6] )   # step 4
  COEX( v[3], v[5] );  COEX( v[2], v[4] )   # step 5
  COEX( v[3], v[4] );                       # step 6
  COEX( v[3], v[8] );                       # step 7
  COEX( v[4], v[8] );                       # step 8

  sort8 v[4]    # sort the eight medians in v[4]

  result = average(       # compute the next pivot
    v[4].mm256_extract_epi32 3'i32,
    v[4].mm256_extract_epi32 4'i32
  )
  #debugEcho fmt"getPivot:: new pivot-{result}"


# recursion for quicksort
#
proc qsCore( arrPtr :pointer, left, right :int32; chooseAvg :var bool = false; avg :int32 = 0 ) =

  let arr = cast[ptr UncheckedArray[int32]]( arrPtr )

  if right - left < 513 :
    #
    # use a sorting-network for tiny arrays with less than 512-elements.

    # buffer for sorting-networks
    #
    var buffer :array[66, M256i]
    sortInt32Network(
      arr[left].addr,
      buffer[0].addr,
      right - left + 1
    )
    return

  # avg is average of largest- and smallest-values in array.
  #
  var
    pivot    = if chooseAvg: avg else: arr.getPivot( left, right )
    smallest = int32.high   # smallest value after partitioning
    biggest  = int32.low    # largest value after partitioning
    bound    = arr.partitionVectorized64( left, right.succ, pivot, smallest.addr, biggest.addr )

    # the ratio of the length of the smaller partition to the array length.
    #
    ratio :float = min( right - bound.pred, bound - left ) / ( right - left.succ )


  # In case of unbalanced sub-arrays, change the pivot-selection strategy.
  #
  if ratio < 0.2 : chooseAvg = not chooseAvg
  if pivot != smallest : # if values in the left sub-array are not identical
    arr.qsCore( left, bound.pred, chooseAvg, smallest.average pivot )
  if pivot.succ != biggest: # if values in the right sub-array are not identical
    arr.qsCore( bound, right, chooseAvg, biggest.average  pivot )


# call this function for sorting.
#
proc quicksort*[T :int32]( loc :ptr T, n :int ) =
  var tf :bool
  if loc[].sizeof == 4:
    #echo fmt"qs: sorting {n} x {$T} of length-{T.sizeof}"
    qsCore( loc, 0'i32, n.int32-1, tf )

when isMainModule:
  echo "nothing here, compile using -d:app"