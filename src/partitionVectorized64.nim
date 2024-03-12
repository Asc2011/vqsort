
# simulate wider vector registers to speedup sorting
#
proc partitionVectorized64( arr :ptr UncheckedArray[int32], le, ri, pi :int32, smallest, biggest :ptr int32 ) :int32 =

  var (left, right, pivot) = ( le, ri, pi )

  # do not optimize if less than 129 elements.
  #
  if right - left < 129 :
    return arr.partitionVectorized8( left, right, pivot, smallest, biggest )


  # make array length divisible by eight, shortening the array.
  #
  for i in countdown( (right-left) mod 8, 1, 1 ) :
    smallest[]  = smallest[].min arr[left]
    biggest[]   =  biggest[].max arr[left]

    if arr[left] > pivot :
      right.dec
      arr[left].swap arr[right]
    else : left.inc

  var
    pivotVec = mm256_set1_epi32 pivot      # fill vector with pivot
    sv       = mm256_set1_epi32 smallest[] # vector for smallest elements
    bv       = mm256_set1_epi32 biggest[]  # vector for biggest elements

  # make array length divisible by 64, shortening the array
  #
  for i in countdown( ((right-left) mod 64) div 8, 1, 1 ) :

    let vecL   = load arr[left].addr

    sv = vecL.mm256_min_epi32 sv
    bv = vecL.mm256_max_epi32 bv

    let
      cmpVec       = vecL.mm256_cmpgt_epi32 pivotVec
      mm           = mm256_movemask_ps( mm256_castsi256_ps cmpVec )
      countGtPivot = mm_popcnt_u32 mm.uint32
      permuteVec   = vecL.mm256_permutevar8x32_epi32 load permutationMasks[mm*8].addr

      # this is a slower way to partition an array with vector instructions
      #
      maskVec    = permuteVec.mm256_cmpgt_epi32 pivotVec
      vecR       = load arr[right - 8].addr
      vecLnew    = permuteVec.mm256_blendv_epi8( vecR, maskVec )
      vecRnew    = vecR.mm256_blendv_epi8( permuteVec, maskVec )

    store( arr[left].addr,    vecLnew )
    store( arr[right-8].addr, vecRnew )
    left.inc ( 8 - countGtPivot )
    right.dec countGtPivot

  # buffer 8-vectors from both sides of the array
  #
  var vecLeft1  = load arr[ left   ].addr  ; var vecLeft2  = load arr[ left+ 8].addr
  var vecLeft3  = load arr[ left+16].addr  ; var vecLeft4  = load arr[ left+24].addr
  var vecLeft5  = load arr[ left+32].addr  ; var vecLeft6  = load arr[ left+40].addr
  var vecLeft7  = load arr[ left+48].addr  ; var vecLeft8  = load arr[ left+56].addr
  var vecRight1 = load arr[right-64].addr  ; var vecRight2 = load arr[right-56].addr
  var vecRight3 = load arr[right-48].addr  ; var vecRight4 = load arr[right-40].addr
  var vecRight5 = load arr[right-32].addr  ; var vecRight6 = load arr[right-24].addr
  var vecRight7 = load arr[right-16].addr  ; var vecRight8 = load arr[right- 8].addr


  # store points of the vectors
  #
  var
    rightIdx = right - 64'i32 # right store-point
    leftIdx  = left           # left stor-point

  # indices for loading the elements
  #
  left.inc  64'i32 # increase because first 64 elements are cached
  right.dec 64'i32 # decrease because last 64 elements are cached

  var
    countGTpivot1, countGTpivot2, countGTpivot3, countGTpivot4 :int32
    countGTpivot5, countGTpivot6, countGTpivot7, countGTpivot8 :int32

  while (right - left) != 0 :
    #
    # partition 64-elements per iteration.
    #
    var currVec1, currVec2, currVec3, currVec4 :M256i
    var currVec5, currVec6, currVec7, currVec8 :M256i

    # if less elements are stored on the right-side of the array,
    # then next 8-vectors load from the right-side, otherwise load from the left-side.
    #
    if (rightIdx + 64) - right < (left - leftIdx) :
      right.dec 64'i32
      currVec1 = load arr[right   ].addr ;  currVec2 = load arr[right+ 8].addr
      currVec3 = load arr[right+16].addr ;  currVec4 = load arr[right+24].addr
      currVec5 = load arr[right+32].addr ;  currVec6 = load arr[right+40].addr
      currVec7 = load arr[right+48].addr ;  currVec8 = load arr[right+56].addr
    else :
      currVec1 = load arr[left   ].addr
      currVec2 = load arr[left+ 8].addr
      currVec3 = load arr[left+16].addr ;   currVec4 = load arr[left+24].addr
      currVec5 = load arr[left+32].addr ;   currVec6 = load arr[left+40].addr
      currVec7 = load arr[left+48].addr ;   currVec8 = load arr[left+56].addr
      left.inc 64'i32

    # partition 8-vectors and store them on both sides of the array
    #
    countGTpivot1 = partitionVec( currVec1, pivotVec, sv, bv )
    countGTpivot2 = partitionVec( currVec2, pivotVec, sv, bv )
    countGTpivot3 = partitionVec( currVec3, pivotVec, sv, bv )
    countGTpivot4 = partitionVec( currVec4, pivotVec, sv, bv )
    countGTpivot5 = partitionVec( currVec5, pivotVec, sv, bv )
    countGTpivot6 = partitionVec( currVec6, pivotVec, sv, bv )
    countGTpivot7 = partitionVec( currVec7, pivotVec, sv, bv )
    countGTpivot8 = partitionVec( currVec8, pivotVec, sv, bv )

    store( arr[leftIdx].addr, currVec1 ); leftIdx.inc ( 8 - countGTpivot1 )
    store( arr[leftIdx].addr, currVec2 ); leftIdx.inc ( 8 - countGTpivot2 )
    store( arr[leftIdx].addr, currVec3 ); leftIdx.inc ( 8 - countGTpivot3 )
    store( arr[leftIdx].addr, currVec4 ); leftIdx.inc ( 8 - countGTpivot4 )
    store( arr[leftIdx].addr, currVec5 ); leftIdx.inc ( 8 - countGTpivot5 )
    store( arr[leftIdx].addr, currVec6 ); leftIdx.inc ( 8 - countGTpivot6 )
    store( arr[leftIdx].addr, currVec7 ); leftIdx.inc ( 8 - countGTpivot7 )
    store( arr[leftIdx].addr, currVec8 ); leftIdx.inc ( 8 - countGTpivot8 )

    store( arr[rightIdx+56].addr, currVec1 ); rightIdx.dec countGTpivot1
    store( arr[rightIdx+56].addr, currVec2 ); rightIdx.dec countGTpivot2
    store( arr[rightIdx+56].addr, currVec3 ); rightIdx.dec countGTpivot3
    store( arr[rightIdx+56].addr, currVec4 ); rightIdx.dec countGTpivot4
    store( arr[rightIdx+56].addr, currVec5 ); rightIdx.dec countGTpivot5
    store( arr[rightIdx+56].addr, currVec6 ); rightIdx.dec countGTpivot6
    store( arr[rightIdx+56].addr, currVec7 ); rightIdx.dec countGTpivot7
    store( arr[rightIdx+56].addr, currVec8 ); rightIdx.dec countGTpivot8


  # partition and store 8-vectors coming from the left-side of the array.
  #
  countGTpivot1 = partitionVec( vecLeft1, pivotVec, sv, bv )
  countGTpivot2 = partitionVec( vecLeft2, pivotVec, sv, bv )
  countGTpivot3 = partitionVec( vecLeft3, pivotVec, sv, bv )
  countGTpivot4 = partitionVec( vecLeft4, pivotVec, sv, bv )
  countGTpivot5 = partitionVec( vecLeft5, pivotVec, sv, bv )
  countGTpivot6 = partitionVec( vecLeft6, pivotVec, sv, bv )
  countGTpivot7 = partitionVec( vecLeft7, pivotVec, sv, bv )
  countGTpivot8 = partitionVec( vecLeft8, pivotVec, sv, bv )

  store( arr[leftIdx].addr, vecLeft1 );  leftIdx.inc ( 8 - countGTpivot1 )
  store( arr[leftIdx].addr, vecLeft2 );  leftIdx.inc ( 8 - countGTpivot2 )
  store( arr[leftIdx].addr, vecLeft3 );  leftIdx.inc ( 8 - countGTpivot3 )
  store( arr[leftIdx].addr, vecLeft4 );  leftIdx.inc ( 8 - countGTpivot4 )
  store( arr[leftIdx].addr, vecLeft5 );  leftIdx.inc ( 8 - countGTpivot5 )
  store( arr[leftIdx].addr, vecLeft6 );  leftIdx.inc ( 8 - countGTpivot6 )
  store( arr[leftIdx].addr, vecLeft7 );  leftIdx.inc ( 8 - countGTpivot7 )
  store( arr[leftIdx].addr, vecLeft8 );  leftIdx.inc ( 8 - countGTpivot8 )

  store( arr[rightIdx+56].addr, vecLeft1 );  rightIdx.dec countGTpivot1
  store( arr[rightIdx+56].addr, vecLeft2 );  rightIdx.dec countGTpivot2
  store( arr[rightIdx+56].addr, vecLeft3 );  rightIdx.dec countGTpivot3
  store( arr[rightIdx+56].addr, vecLeft4 );  rightIdx.dec countGTpivot4
  store( arr[rightIdx+56].addr, vecLeft5 );  rightIdx.dec countGTpivot5
  store( arr[rightIdx+56].addr, vecLeft6 );  rightIdx.dec countGTpivot6
  store( arr[rightIdx+56].addr, vecLeft7 );  rightIdx.dec countGTpivot7
  store( arr[rightIdx+56].addr, vecLeft8 );  rightIdx.dec countGTpivot8

  # partition and store 8-vectors coming from the right-side of the array.
  #
  countGtpivot1 = partitionVec( vecRight1, pivotVec, sv, bv )
  countGtpivot2 = partitionVec( vecRight2, pivotVec, sv, bv )
  countGtpivot3 = partitionVec( vecRight3, pivotVec, sv, bv )
  countGtpivot4 = partitionVec( vecRight4, pivotVec, sv, bv )
  countGtpivot5 = partitionVec( vecRight5, pivotVec, sv, bv )
  countGtpivot6 = partitionVec( vecRight6, pivotVec, sv, bv )
  countGtpivot7 = partitionVec( vecRight7, pivotVec, sv, bv )
  countGtpivot8 = partitionVec( vecRight8, pivotVec, sv, bv )

  store( arr[leftIdx].addr, vecRight1 );  leftIdx.inc ( 8 - countGTpivot1 )
  store( arr[leftIdx].addr, vecRight2 );  leftIdx.inc ( 8 - countGTpivot2 )
  store( arr[leftIdx].addr, vecRight3 );  leftIdx.inc ( 8 - countGTpivot3 )
  store( arr[leftIdx].addr, vecRight4 );  leftIdx.inc ( 8 - countGTpivot4 )
  store( arr[leftIdx].addr, vecRight5 );  leftIdx.inc ( 8 - countGTpivot5 )
  store( arr[leftIdx].addr, vecRight6 );  leftIdx.inc ( 8 - countGTpivot6 )
  store( arr[leftIdx].addr, vecRight7 );  leftIdx.inc ( 8 - countGTpivot7 )
  store( arr[leftIdx].addr, vecRight8 );  leftIdx.inc ( 8 - countGTpivot8 )

  store( arr[rightIdx+56].addr, vecRight1 );  rightIdx.dec countGTpivot1
  store( arr[rightIdx+56].addr, vecRight2 );  rightIdx.dec countGTpivot2
  store( arr[rightIdx+56].addr, vecRight3 );  rightIdx.dec countGTpivot3
  store( arr[rightIdx+56].addr, vecRight4 );  rightIdx.dec countGTpivot4
  store( arr[rightIdx+56].addr, vecRight5 );  rightIdx.dec countGTpivot5
  store( arr[rightIdx+56].addr, vecRight6 );  rightIdx.dec countGTpivot6
  store( arr[rightIdx+56].addr, vecRight7 );  rightIdx.dec countGTpivot7
  store( arr[rightIdx+56].addr, vecRight8 );

  smallest[] = calcMin sv
  biggest[]  = calcMax bv
  result     = leftIdx
