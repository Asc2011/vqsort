# sort 8-columns, each containing 16 x int32, with Green's 60-modules network
#
func sort16IntByColumn( vecsPtr :ptr int32 ) =

  let vecs = cast[ptr UncheckedArray[M256i] ]( vecsPtr )

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
