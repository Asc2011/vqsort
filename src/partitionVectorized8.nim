
proc partitionVectorized8( arr :ptr UncheckedArray[int32]; le, ri, pi :int32; smallest, biggest :ptr int32 ) :int32 =

  var (left, right, pivot) = ( le, ri, pi )

  # make the array-length divisible by eight, shortening the array
  #
  for i in countdown((right-left) mod 8, 1, 1 ) :
    smallest[] = smallest[].min arr[left]
    biggest[]  =  biggest[].max arr[left]

    if arr[left] > pivot:
      right.dec
      arr[left].swap arr[right]
    else: left.inc

  if left == right : return left # less than 8 elements in the array

  var
    pivotVec = mm256_set1_epi32 pivot       # fill vector with pivot
    sv       = mm256_set1_epi32 smallest[]  # vector for smallest elements
    bv       = mm256_set1_epi32 biggest[]   # vector for biggest elements
    countGTpivot :int32

  if right - left == 8 :        # if 8 elements left after shortening
    var v = load arr[left].addr
    countGTpivot = v.partitionVec( pivotVec, sv, bv )
    store( arr[left].addr, v )
    smallest[] = calcMin sv
    biggest[]  = calcMax bv
    result     = left + ( 8 - countGTpivot )

  # first and last 8 values are partitioned at the end.
  #
  var
    leftVec  = load arr[left].addr     # first 8 values
    rightVec = load arr[right-8].addr  # last 8 values
    currVec :M256i                     # vector to be partitioned

                                 # store points of the vectors
    rightIdx :int32 = right - 8  # right store point
    leftIdx  :int32 = left       # left store point

  # indices for loading the elements
  #
  left.inc  8 # increase, because first 8 elements are cached
  right.dec 8 # decrease, because last 8 elements are cached

  while (right - left) != 0 : # partition 8 elements per iteration
    #[ if fewer elements are stored on the right side of the array,
       then next elements are loaded from the right side,
       otherwise from the left side. ]#

    if (rightIdx + 8) - right < (left - leftIdx) :
      right.dec 8
      currVec = load arr[right].addr
    else:
      currVec = load arr[left].addr
      left.inc 8

    # partition the current vector and save it on both sides of the array.
    #
    countGTpivot = currVec.partitionVec( pivotVec, sv, bv )
    # PERF maybe stream here ?
    store( arr[leftIdx].addr,  currVec )
    store( arr[rightIdx].addr, currVec )

    # update both store points
    #
    rightIdx.dec countGTpivot
    leftIdx.inc (8 - countGTpivot)

  # partition and save leftVec and rightVec
  #
  # PERF maybe stream here
  countGTpivot = leftVec.partitionVec( pivotVec, sv, bv )
  store( arr[leftIdx].addr, leftVec )
  leftIdx.inc ( 8 - countGTpivot )

  countGTpivot = rightVec.partitionVec( pivotVec, sv, bv )
  store( arr[leftIdx].addr, rightVec )
  leftIdx.inc ( 8 - countGTpivot )

  smallest[] = calcMin sv   # determine smallest value in vector
  biggest[]  = calcMax bv   # determine largest value in vector
  result     = leftIdx
