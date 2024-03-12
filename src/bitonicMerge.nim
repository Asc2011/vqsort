# merge N vectors with bitonic merge, N % 2 == 0 and N > 0
# s = 2 means that two vectors are already sorted.
#
func bitonicMerge16( bufPtr :pointer, N :int32, s :static int32 = 2) =

  let vecs = cast[ptr UncheckedArray[M256i]]( bufPtr )

  var t = s*2
  while t < 2*N :
    for l in countup( 0, N.pred, t ) :
      for j in countup( max(l+t-N, 0), (t div 2).pred, 2) :
        reverse vecs[ l+t-1-j ]
        reverse vecs[ l+t-2-j ]
        COEX( vecs[l+j],   vecs[ l+t-1-j ])
        COEX( vecs[l+j+1], vecs[ l+t-2-j ])

    var m = t div 2
    while m > 4 :
      var k :int
      while k < (N - m div 2) :
        let bound :int = min( (k + m div 2), N - (m div 2) )
        var j = k
        while j < bound :
          COEX( vecs[j],   vecs[m div 2 + j]      )
          COEX( vecs[j+1], vecs[m div 2 + j.succ] )
          j.inc 2
        k.inc m
      m = m div 2
      
    for j in countup( 0, (N-2).pred, 4 ) :
      COEX( vecs[j],   vecs[j+2] )
      COEX( vecs[j+1], vecs[j+3] )

    for j in countup( 0, N.pred, 2 ) : COEX( vecs[j], vecs[j.succ] )

    for i in countup( 0, N.pred, 2 ) :
      let j = i.succ
      COEXpermute( vecs[i], 4, 5, 6, 7, 0, 1, 2, 3 )
      COEXpermute( vecs[j], 4, 5, 6, 7, 0, 1, 2, 3 )
      var tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32  vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32  vecs[j]
      COEX( vecs[i], vecs[j] )
      tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32  vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32  vecs[j]
      COEX( vecs[i], vecs[j] )
      tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32  vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32  vecs[j]

    t *= 2 # EOL-while t < 2*N


func bitonicMerge128( bufferPtr :pointer, N :int32, s :static int32 = 16) =

  #debugEcho fmt"bitonicMerge128 :: N-{N} s-{s}"

  let
    remainder8  = N - N mod 8
    remainder16 = N - N mod 16

    vecs = cast[ptr UncheckedArray[M256i]]( bufferPtr )

  var t = s*2
  while t < 2*N :
    for l in countup( 0, N.pred, t ) :
      for j in countup( max( l+t-N, 0), (t div 2).pred, 2 ) :
        reverse vecs[ l+t-1-j ]
        reverse vecs[ l+t-2-j ]
        COEX( vecs[l+j],   vecs[ l+t-1-j ] )
        COEX( vecs[l+j+1], vecs[ l+t-2-j ] )

    var m = t div 2
    while m > 16 :
      for k in countup( 0, (N-m).pred, m ) :
        let bound = min( (k + m div 2), N - (m div 2))
        for j in countup( k, bound.pred, 2 ) :
          COEX( vecs[j],   vecs[m div 2+j  ] )
          COEX( vecs[j+1], vecs[m div 2+j+1] )
      m = m div 2 # EOL-while m > 16

    for j in countup( 0, remainder16.pred, 16 ) :
      COEX( vecs[j  ], vecs[j+ 8] )
      COEX( vecs[j+1], vecs[j+ 9] )
      COEX( vecs[j+2], vecs[j+10] )
      COEX( vecs[j+3], vecs[j+11] )
      COEX( vecs[j+4], vecs[j+12] )
      COEX( vecs[j+5], vecs[j+13] )
      COEX( vecs[j+6], vecs[j+14] )
      COEX( vecs[j+7], vecs[j+15] )

    for j in countup( remainder16+8, N.pred, 1 ) : COEX( vecs[j-8], vecs[j] )

    for j in countup( 0, remainder8.pred, 8 ) :
      COEX( vecs[j  ], vecs[j+4] )
      COEX( vecs[j+1], vecs[j+5] )
      COEX( vecs[j+2], vecs[j+6] )
      COEX( vecs[j+3], vecs[j+7] )

    for j in (remainder8+4) ..< N : COEX( vecs[j-4], vecs[j] )

    for j in countup( 0, (N-2).pred, 4 ) :
      COEX( vecs[j  ], vecs[j+2] )
      COEX( vecs[j+1], vecs[j+3] )

    for j in countup( 0, N.pred, 2) : COEX( vecs[j], vecs[j+1] )

    for i in countup( 0, N.pred, 2) :
      let j = i.succ
      COEXpermute( vecs[i], 4, 5, 6, 7, 0, 1, 2, 3 )
      COEXpermute( vecs[j], 4, 5, 6, 7, 0, 1, 2, 3 )
      var tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32 vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32 vecs[j]
      COEX( vecs[i], vecs[j] )
      tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32 vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32 vecs[j]
      COEX( vecs[i], vecs[j] )
      tempVec = vecs[i]
      vecs[i] = vecs[i].mm256_unpacklo_epi32 vecs[j]
      vecs[j] = tempVec.mm256_unpackhi_epi32 vecs[j]

    t *= 2 # EOL-while t < 2*N
