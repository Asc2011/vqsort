{.experimental: "callOperator".}

import std/[strformat, strutils, bitops]
import nimsimd/[ avx2, sse2,bmi1 ]

when defined(gcc) or defined(clang):
  {.localPassc: "-mavx2 -mavx -msse2".}


func tzu32*( i :uint32 ) :int = mm_tzcnt_32( i )

export avx2

type UArr[T] = UncheckedArray[T]

type
  epi8*    = int8 | char
  epi16*   = int16
  epi32*   = int32
  epi64*   = int | int64
  epi128*  = int64

  epu8*    = uint8 | byte
  epu16*   = uint16
  epu32*   = uint32
  epu64*   = uint | uint64
  epu128*  = uint64

  ps*      = float32
  pd*      = float64

type epu* = epu8 | epu16 | epu32 | epu64 | byte # | epu128
type epi* = epi8 | epi16 | epi32 | epi64 | char | epi128

type SomeSIMD* = epi | epu | ps | pd

#type V128*      = M128 | M128d | M128i
#type V256*      = M256 | M256d | M256i
#type V512*      = M512 | M512d | M512i

type SomeVec*   = V128 | V256

proc `()`*( vec :SomeVec, tt :typedesc[SomeSIMD] ) :string

func sizeof*( tt :typedesc[SomeSIMD] ) :int =
  when (tt is epi8 ) or (tt is epu8) : 1
  elif (tt is epi16) or (tt is epu16): 2
  elif (tt is epi32) or (tt is epu32) or (tt is ps): 4
  elif (tt is epi64) or (tt is epu64) or (tt is pd): 8
  else: 0

func sizeof[V :SomeVec]( vec :V ) :int =
  when V is V256: 32
  elif V is M128: 16
  else: 8


func csb64*( v :uint | uint64 ) :int = mm_popcnt_u64 v
func csb32*( v :uint32 )        :int = mm_popcnt_u32 v
func csb*( v :SomeInteger )     :int =
  ## count set bits
  when v is SomeSignedInt:
    when (v is int) or (v is int64):   csb64( v.uint64 )
    else: csb32( v.uint32 )
  elif v is SomeUnsignedInt:
    when (v is uint) or (v is uint64): v.csb64
    else: v.csb32

# cannot import these ?
# func csb64*( v :int | int64 ) :int = popcnt64( v )
# func csb32*( v :int32 ) :int = popcnt32( v )

func mallocAligned*( bytes, align :int ) :pointer = mm_malloc( bytes, align )

func asM128*(  vec :M128i ) :M128  = vec.mm_castsi128_ps()
func asM128d*( vec :M128i ) :M128d = vec.mm_castsi128_pd()
func asM128i*( vec :M128 )  :M128i = vec.mm_castps_si128()
func asM128i*( vec :M256i ) :M128i = vec.mm256_castsi256_si128()

func asM256i*( vec :M256d ) :M256i = vec.mm256_castpd_si256()
func asM256i*( vec :M256i ) :M256i = vec
func asM256i*( vec :M128i ) :M256i = vec.mm256_castsi128_si256()
func asM256i*( vec :M256  ) :M256i = vec.mm256_castps_si256()
func asM256d*( vec :M256i ) :M256d = vec.mm256_castsi256_pd()
func asM256*(  vec :M256i ) :M256  = vec.mm256_castsi256_ps()

func alignr*[V :V256]( vecA, vecB :V, imm8 :static int ) :V =
  when V is M256i: mm256_alignr_epi8( vecA, vecB, imm8.int32 )
  elif V is M256:  mm256_alignr_epi8( vecA.asM256i, vecB.asM256i, imm8.int32 ).asM256
  elif V is M256d: mm256_alignr_epi8( vecA.asM256i, vecB.asM256i, imm8.int32 ).asM256d


# logical operators

func `and`*[V :SomeVec]( vecA, vecB :V  ) :V =
  when V is M256i: vecA.mm256_and_si256 vecB
  elif V is M256:  vecA.mm256_and_ps vecB
  elif V is M256d: vecA.mm256_and_pd vecB
  elif V is M128i: vecA.mm_and_si128 vecB
  elif V is M128:  vecA.mm_and_ps vecB
  elif V is M128d: vecA.mm_and_pd vecB

func `or`*[V :SomeVec]( vecA, vecB :V   ) :V =
  when V is M256i: vecA.mm256_or_si256 vecB
  elif V is M256:  vecA.mm256_or_ps vecB
  elif V is M256d: vecA.mm256_or_pd vecB
  elif V is M128i: vecA.mm_or_si128 vecB
  elif V is M128:  vecA.mm_or_ps vecB
  elif V is M128d: vecA.mm_or_pd vecB

func `xor`*[V :SomeVec]( vecA, vecB :V  ) :V =
  when V is M256i: vecA.mm256_xor_si256 vecB
  elif V is M256:  vecA.mm256_xor_ps vecB
  elif V is M256d: vecA.mm256_xor_pd vecB
  elif V is M128i: vecA.mm_xor_si128 vecB
  elif V is M128:  vecA.mm_xor_ps vecB
  elif V is M128d: vecA.mm_xor_pd vecB

func andnot*[V :SomeVec]( vecA, vecB :V ) :V =
  when V is M256i: vecA.mm256_andnot_si256 vecB
  elif V is M256:  vecA.mm256_andnot_ps vecB
  elif V is M256d: vecA.mm256_andnot_pd vecB
  elif V is M128i: vecA.mm_andnot_si128 vecB
  elif V is M128:  vecA.mm_andnot_ps vecB
  elif V is M128d: vecA.mm_andnot_pd vecB

# math

func add*[V :SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi64: vecA.mm256_add_epi64 vecB
    elif tt is epi32: vecA.mm256_add_epi32 vecB
    elif tt is epi16: vecA.mm256_add_epi16 vecB
    elif tt is epi8:  vecA.mm256_add_epi8  vecB
  elif V is M256:  vecA.mm256_add_ps vecB
  elif V is M256d: vecA.mm256_add_pd vecB
  elif V is M128i:
    when tt is epi64: vecA.mm_add_epi64 vecB
    elif tt is epi32: vecA.mm_add_epi32 vecB
    elif tt is epi16: vecA.mm_add_epi16 vecB
    elif tt is epi8:  vecA.mm_add_epi8  vecB
  elif V is M128:  vecA.mm_add_ps vecB
  elif V is M128d: vecA.mm_add_pd vecB

func sub*[V :SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi64: vecA.mm256_sub_epi64 vecB
    elif tt is epi32: vecA.mm256_sub_epi32 vecB
    elif tt is epi16: vecA.mm256_sub_epi16 vecB
    elif tt is epi8:  vecA.mm256_sub_epi8  vecB
  elif V is M256:  vecA.mm256_sub_ps vecB
  elif V is M256d: vecA.mm256_sub_pd vecB
  elif V is M128i:
    when tt is epi64: vecA.mm_sub_epi64 vecB
    elif tt is epi32: vecA.mm_sub_epi32 vecB
    elif tt is epi16: vecA.mm_sub_epi16 vecB
    elif tt is epi8:  vecA.mm_sub_epi8  vecB
  elif V is M128:  vecA.mm_sub_ps vecB
  elif V is M128d: vecA.mm_sub_pd vecB

func subs*[V :M256i]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when tt is epi64: vecA.mm256_subs_epi64 vecB
  elif tt is epi32: vecA.mm256_subs_epi32 vecB
  elif tt is epi16: vecA.mm256_subs_epi16 vecB
  elif tt is epi8:  vecA.mm256_subs_epi8  vecB

func adds*[V :M256i]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when tt is epi16: vecA.mm256_subs_epi16 vecB
  elif tt is epu16: vecA.mm256_subs_epu16 vecB
  elif tt is epi8:  vecA.mm256_subs_epi8  vecB
  elif tt is epu8:  vecA.mm256_subs_epu8  vecB

func mul*[V :SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi32 ) :V =
  when V is M256i:
    when tt is epi32: vecA.mm256_mul_epi32 vecB
    elif tt is epu32: vecA.mm256_mul_epu32 vecB
  elif V is M256:  vecA.mm256_mul_ps vecB
  elif V is M256d: vecA.mm256_mul_pd vecB
  elif V is M128i:
    when tt is epi32: vecA.mm_mul_epi32 vecB
    elif tt is epu32: vecA.mm_mul_epu32 vecB
  elif V is M128:  vecA.mm_mul_ps vecB
  elif V is M128d: vecA.mm_mul_pd vecB

func `div`*[V :SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi64: mm256_div_ps( vecA.asM256d, vecB.asM256d ).asM256i
    elif tt is epi32: mm256_div_ps( vecA.asM256, vecB.asM256 ).asM256i
    #elif tt is epu32: vecA.mm256_mul_epu32 vecB
  elif V is M256:  vecA.mm256_div_ps vecB
  elif V is M256d: vecA.mm256_div_pd vecB
  elif V is M128i:
    when tt is epi64: mm_mul_pd( vecA.asM128d, vecB.asM128d ).asM128i
    elif tt is epi32: mm_mul_ps( vecA.asM128, vecB.asM128 ).asM128i
    #elif tt is epu32: vecA.mm_mul_epu32 vecB
  elif V is M128:  vecA.mm_div_ps vecB
  elif V is M128d: vecA.mm_div_pd vecB

func abs*[V :M256i]( vecA :V, tt :typedesc[epi] = epi32 ) :M256i =
  when tt is epi32: mm256_abs_epi32 vecA
  elif tt is epi16: mm256_abs_epi16 vecA
  elif tt is epi8:  mm256_abs_epi8  vecA

func abs*[V :M128i]( vecA :V, tt :typedesc[epi] = epi32 ) :M128i =
  when tt is epi32: mm_abs_epi32 vecA
  elif tt is epi16: mm_abs_epi16 vecA
  elif tt is epi8:  mm_abs_epi8  vecA

func avg*[V :M256i]( vecA :V, tt :typedesc[epi] = epu16 ) :M256i =
  when tt is epi16: mm256_avg_epu16 vecA
  elif tt is epi8:  mm256_avg_epu8  vecA

func avg*[V :M128i]( vecA :V, tt :typedesc[epi] = epu16 ) :M128i =
  when tt is epi16: mm_avg_epu16 vecA
  elif tt is epi8:  mm_avg_epu8  vecA

func sign*[V :M256i]( vecA, vecB :V, tt :typedesc[epi] = epi32 ) :M256i =
  when tt is epi32: mm256_sign_epi32( vecA, vecB )
  elif tt is epi16: mm256_sign_epi16( vecA, vecB )
  elif tt is epi8:  mm256_sign_epi8(  vecA, vecB )

func sign*[V :M128i]( vecA, vecB :V, tt :typedesc[epi] = epi32 ) :M128i =
  when tt is epi32: mm_sign_epi32( vecA, vecB )
  elif tt is epi16: mm_sign_epi16( vecA, vecB )
  elif tt is epi8:  mm_sign_epi8(  vecA, vecB )



# bitwise operations

func `shr`*[V :SomeVec]( vec :V, imm8 :int32, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when (tt is epi64) or (tt is epu64) : vec.mm256_srli_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm256_srli_epi32 imm8
    elif (tt is epi16) or (tt is epu16) : vec.mm256_srli_epi16 imm8
  elif V is M256:  vec.asM256i.shr(imm8, epi32)
  elif V is M256d: vec.asM256i.shr(imm8, epi64)
  elif V is M128i:
    when (tt is epi64) or (tt is epu64) : vec.mm_srli_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm_srli_epi32 imm8
    elif (tt is epi16) or (tt is epu16) : vec.mm_srli_epi16 imm8
  elif V is M128:  vec.asM256i.shr( imm8, epi32 )
  elif V is M128d: vec.asM256i.shr( imm8, epi64 )

func shiftRight*[V :SomeVec]( vec :V, imm8 :int32, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shr`( imm8, tt )

func `shr`*[V :V256]( vec :V, count :M256i, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when (tt is epi64) or (tt is epu64) : vec.mm256_srliv_epi64 count
    elif (tt is epi32) or (tt is epu32) : vec.mm256_srliv_epi32 count
  elif V is M256:  vec.asM256i.shr(count, epi32).asM256
  elif V is M256d: vec.asM256i.shr(count, epi64).asM256d

func `shr`*[V :V128]( vec :V, count :M128i, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M128i:
    when (tt is epi64) or (tt is epu64) : vec.mm_srliv_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm_srliv_epi32 imm8
  elif V is M128:  vec.asM128i.shr( count, epi32 ).asM128
  elif V is M128d: vec.asM128i.shr( count, epi64 ).asM128d

func shiftRight*[V :V256]( vec :V, count :int32, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shr`( count, tt )
func shiftRight*[V :V128]( vec :V, count :int32, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shr`( count, tt )

func `shl`*[V :SomeVec]( vec :V, imm8 :int32, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when (tt is epi64) or (tt is epu64) : vec.mm256_slli_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm256_slli_epi32 imm8
    elif (tt is epi16) or (tt is epu16) : vec.mm256_slli_epi16 imm8
  elif V is M256:  vec.asM256i.shl(imm8, epi32).asM256
  elif V is M256d: vec.asM256i.shl(imm8, epi64).asM256d
  elif V is M128i:
    when (tt is epi64) or (tt is epu64) : vec.mm_slli_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm_slli_epi32 imm8
    elif (tt is epi16) or (tt is epu16) : vec.mm_slli_epi16 imm8
  elif V is M128:  vec.asM128i.shl( imm8, epi32 ).asM128
  elif V is M128d: vec.asM128i.shl( imm8, epi64 ).asM128d

func shiftLeft*[V :SomeVec]( vec :V, imm8 :int32, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shl`( imm8, tt )

func `shl`*[V :V256]( vec :V, count :M256i, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when (tt is epi64) or (tt is epu64) : vec.mm256_slliv_epi64 count
    elif (tt is epi32) or (tt is epu32) : vec.mm256_slliv_epi32 count
  elif V is M256:  vec.asM256i.shl(count, epi32).asM256
  elif V is M256d: vec.asM256i.shl(count, epi64).asM256d

func `shl`*[V :V128]( vec :V, count :M128i, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M128i:
    when (tt is epi64) or (tt is epu64) : vec.mm_slliv_epi64 imm8
    elif (tt is epi32) or (tt is epu32) : vec.mm_slliv_epi32 imm8
  elif V is M128:  vec.asM128i.shl( count, epi32 ).asM128
  elif V is M128d: vec.asM128i.shl( count, epi64 ).asM128d

func shiftLeft*[V :V256]( vec :V, count :M256i, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shl`( count, tt )
func shiftLeft*[V :V128]( vec :V, count :M128i, tt :typedesc[SomeSIMD] = epi64 ) :V = vec.`shl`( count, tt )


func byteShift*[V :V256]( vec :V, imm8 :static int32, tt :typedesc[epi128] = epi128 ) :V =
  when V is M256i: vec.mm256_slli_epi128 imm8
  elif V is M256:  vec.asM256i.mm256_slli_epi128( imm8 ).asM256
  elif V is M256d: vec.asM256i.mm256_slli_epi128( imm8 ).asM256d

func byteShift*[V :V128]( vec :V, imm8 :int32 ) :V =
  when V is M128i: vec.mm_bslli_si128 imm8
  elif V is M128:  vec.asM128i.byteShift( imm8 ).asM128
  elif V is M128d: vec.asM128i.byteShift( imm8 ).asM128d




func blend*[V :SomeVec]( vecA, vecB :V; imm8 :static int32, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256d:
    when tt is epi128: mm256_permute2f128_pd( vecA, vecB, imm8 )
    elif tt is pd:     mm256_blend_pd( vecA, vecB, imm8 )
  elif V is M256:
    when tt is epi128: mm256_permute2f128_ps( vecA, vecB, imm8 )
    elif tt is ps:     mm256_blend_ps( vecA, vecB, imm8 )
  elif V is M256i:
    when tt is epi128: mm256_permute2f128_si256( vecA, vecB, imm8 )
    elif tt is epi64:  mm256_blend_pd( vecA.asM256d, vecB.asM256d, imm8 ).asM256i
    elif tt is epi32:  mm256_blend_epi32( vecA, vecB, imm8 )
    elif tt is epi16:  mm256_blend_epi16( vecA, vecB, imm8 ) #! lanes!

func blend*[V :V256]( vecA, vecB, mask :V, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256d: mm256_blendv_pd(   vecA, vecB, mask )
  elif V is M256i: mm256_blendv_epi8( vecA, vecB, mask )
  elif V is M256:  mm256_blendv_ps(   vecA, vecB, mask )

func blend*[V :V128]( vecA, vecB, mask :V ) :V =
  when V is M128d: mm_blendv_pd(   vecA, vecB, mask )
  elif V is M128i: mm_blendv_epi8( vecA, vecB, mask )
  elif V is M128:  mm_blendv_ps(   vecA, vecB, mask )


func shuffle*[V :V128]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi8 ) :V =
  when V is M128i:
    when tt is epi32: vecA.mm_shuffle_epi32 vecB
    elif tt is epi8:  vecA.mm_shuffle_epi8 vecB
  elif V is ps: vecA.mm_shuffle_ps vecB
  elif V is pd: vecA.mm_shuffle_pd vecB

func shuffle*[V :V256]( vec :var V, imm8 :static int32 ) =
  when V is avx.M256i: vec = vec.mm256_permute4x64_epi64 imm8
  elif V is avx.M256d: vec = vec.mm256_permute4x64_pd    imm8
  else: debugEcho "nimsimd/common_avx2.nim :: This shuffle is not yet possible."

func shuffleLanes*[V : V256]( vec :V, imm8 :static int32, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is avx.M256i:
    when tt is epi64: vec.asM256d.mm256_permute_pd( imm8 ).asM256i
    elif tt is epi32: vec.mm256_shuffle_epi32 imm8
  elif V is avx.M256d: vec.mm256_permute_pd imm8
  elif V is avx.M256:  vec.mm256_permute_ps imm8
  else: debugEcho "nimsimd/common_avx2.nim :: This shuffle is not yet possible."

func shuffleLanes*[V :V256]( vecA, vecB :V; imm8 :static int32, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is avx.M256i:
    when tt is epi64:  mm256_shuffle_pd( vecA.asM256d, vecB.asM256d, imm8 ).asM256i
    elif tt is epi128: mm256_permute2f128_si256( vecA, vecB, imm8 )
    elif tt is epi32:  mm256_shuffle_ps( vecA.asM256,vecB.asM256, imm8 ).asM256i
  elif V is avx.M256d: mm256_shuffle_pd( vecA, vecB, imm8 )
  elif V is avx.M256:  mm256_shuffle_ps( vecA, vecB, imm8 )
  else: debugEcho "nimsimd/common_avx2.nim :: This shuffle is not yet possible."


func moveMask*[V: SomeVec]( vec :V, tt :typedesc[SomeSIMD] = epi8 ) :uint32 =
  # TODO moveMask for epi16, float16
  var v: int32
  when V is M256d: v = vec.mm256_movemask_pd()
  elif V is M256:  v = vec.mm256_movemask_ps()
  elif V is M256i:
    when tt is epi64: v = vec.asM256d.mm256_movemask_pd()
    elif tt is epi32: v = vec.asM256.mm256_movemask_ps()
    else: v = vec.mm256_movemask_epi8()
  else: v = mm256_movemask_epi8( vec.asM256i )
  result = v.uint32

func cmp*[V: M256d | M256]( vecA, vecB :V, pred :static int ) :V =
  when V is M256d: mm256_cmp_pd( vecA, vecB, pred.int32 )
  elif V is M256:  mm256_cmp_ps( vecA, vecB, pred.int32 )

func cmpEq*[V: M256i | M128i]( vecA, vecB :V, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi64: vecA.mm256_cmpeq_epi64 vecB
    elif tt is epu64: vecA.mm256_cmpeq_epi64 vecB
    elif tt is epi32: vecA.mm256_cmpeq_epi32 vecB
    elif tt is epi16: vecA.mm256_cmpeq_epi16 vecB
    elif tt is epi8:  vecA.mm256_cmpeq_epi8  vecB
    elif tt is epu8:  vecA.mm256_cmpeq_epi8  vecB
  elif V is M128i:
    when tt is epi64: vecA.mm_cmpeq_epi64 vecB
    elif tt is epi32: vecA.mm_cmpeq_epi32 vecB
    elif tt is epi16: vecA.mm_cmpeq_epi16 vecB
    elif tt is epi8 : vecA.mm_cmpeq_epi8  vecB

func `===`*[V : SomeVec]( vecA, vecB :V ) :V = vecA.cmpEq vecB

func cmpGt*[V: M256i | M128i]( vecA, vecB :V, tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epu64: vecA.mm256_cmpgt_epi64 vecB
    elif tt is epi64: vecA.mm256_cmpgt_epi64 vecB
    elif tt is epi32: vecA.mm256_cmpgt_epi32 vecB
    elif tt is epi16: vecA.mm256_cmpgt_epi16 vecB
    elif tt is epi8 : vecA.mm256_cmpgt_epi8  vecB
  elif V is M128i:
    when tt is epi64: vecA.mm_cmpgt_epi64 vecB
    elif tt is epi32: vecA.mm_cmpgt_epi32 vecB
    elif tt is epi16: vecA.mm_cmpgt_epi16 vecB


#[
uint8_t _mm256_hmax_index(const __m256i v)
{
    __m256i vmax = v;

    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
    vmax = _mm256_max_epu32(vmax, _mm256_permute2x128_si256(vmax, vmax, 0x01));

    __m256i vcmp = _mm256_cmpeq_epi32(v, vmax);
    uint32_t mask = _mm256_movemask_epi8(vcmp);

    return __builtin_ctz(mask) >> 2;
}
]#

#[
__m256i pmax_epi64(__m256i a, __m256i b)
{
    __m256i mask = _mm256_cmpgt_epi64(a,b);
    return _mm256_blendv_epi8(b,a,mask);
}
 ]#
func pmax*( vecA, vecB :M256i, tt :typedesc[SomeSIMD] = epi64 ) :M256i =
  let mask = cmpGt( vecA, vecB, tt )
  result   = blend( vecB, vecA, mask, tt )

proc pmin*( vecA, vecB :M256i, tt :typedesc[SomeSIMD] = epi64 ) :M256i =
  let mask = cmpGt( vecB, vecA, tt )
  result   = blend( vecB, vecA, mask, tt )

func max_epu64*[V: M256i]( vecA, vecB :V ) :V =
  # https://stackoverflow.com/questions/54394350/simd-implement-mm256-max-epu64-and-mm256-min-epu64
  debugEcho "max_epu64 for \n\t",vecA(epi64),"\n\t",vecB(epi64)
  let
    signbit = mm256_set1_epi64x( 0x8000_0000_0000_0000 )
    mask    = cmpGt(
      ( vecA.xor signbit ),
      ( vecB.xor signbit ),
      epu64
    )
  result = blend( vecB, vecA, mask, epi8 )
  debugEcho "\t",result(epi64)

func min_epu64*[V: M256i]( vecA, vecB :V ) :V =
  debugEcho "min_epu64"
  let
    signbit = mm256_set1_epi64x( 0x8000_0000_0000_0000 )
    mask    = cmpGt(
      ( vecA.xor signbit ),
      ( vecB.xor signbit ),
      epu64
    )
  result = blend( vecB, vecA, mask, epi8 )

func max*[V: SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi32: vecA.mm256_max_epi32 vecB
    elif tt is epi16: vecA.mm256_max_epi16 vecB
    elif tt is epu64: mm256_max_pd( vecA.asM256d, vecB.asM256d ).asM256i
    elif tt is epi64: pmax( vecA, vecB )
    elif tt is epu32: vecA.mm256_max_epu32 vecB
    elif tt is epu16: vecA.mm256_max_epu16 vecB
    elif tt is epu8 : vecA.mm256_max_epu8  vecB
    elif tt is epi8 : vecA.mm256_max_epi8  vecB
    #elif tt is epi64: mm256_max_pd( vecA.asM256d, vecB.asM256d ).asM256i
  elif V is M128i:
    when tt is epi32: vecA.mm_max_epi32 vecB
    elif tt is epi16: vecA.mm_max_epi16 vecB
    elif tt is epi8 : vecA.mm_max_epi8  vecB
    elif tt is epu32: vecA.mm_max_epu32 vecB
    elif tt is epu16: vecA.mm_max_epu16 vecB
    elif tt is epu8 : vecA.mm_max_epu8  vecB
    #elif tt is epu64: max_epu64( vecA, vecB )
    elif tt is epi64: mm_max_pd( vecA, vecB ).asM256i
  elif V is M128:  vecA.mm_max_ps vecB
  elif V is M128d: vecA.mm_max_pd vecB
  elif V is M256:  vecA.mm256_max_ps vecB
  elif V is M256d: vecA.mm256_max_pd vecB

proc min*[V: SomeVec]( vecA, vecB :V; tt :typedesc[SomeSIMD] = epi64 ) :V =
  when V is M256i:
    when tt is epi64: pmin( vecA, vecB )
    elif tt is epu64: mm256_min_pd( vecA.asM256d, vecB.asM256d ).asM256i
    elif tt is epi32: vecA.mm256_min_epi32 vecB
    elif tt is epi16: vecA.mm256_min_epi16 vecB
    elif tt is epu32: vecA.mm256_min_epu32 vecB
    elif tt is epu16: vecA.mm256_min_epu16 vecB
    elif tt is epu8 : vecA.mm256_min_epu8  vecB
    elif tt is epi8 : vecA.mm256_min_epi8  vecB
  elif V is M128i:
    when tt is epi64: mm_min_pd( vecA.asM128d, vecB.asM128d ).asM128i
    elif tt is epi32: vecA.mm_min_epi32 vecB
    elif tt is epi16: vecA.mm_min_epi16 vecB
    elif tt is epi8 : vecA.mm_min_epi8  vecB
    elif tt is epu32: vecA.mm_min_epu32 vecB
    elif tt is epu16: vecA.mm_min_epu16 vecB
    elif tt is epu8 : vecA.mm_min_epu8  vecB
  elif V is M128:  vecA.mm_min_ps vecB
  elif V is M128d: vecA.mm_min_pd vecB
  elif V is M256:  vecA.mm256_min_ps vecB
  elif V is M256d: vecA.mm256_min_pd vecB


#[
func horizontalMinimum*( vec :M256d ) :float64 =

  var i = vec.mm256_extractf128_pd 1'i32
  i = mm_min_pd( i, mm256_castpd256_pd128( vec ) )
  i = mm_min_pd( i, mm_move_sd( i, i ) )
  i = mm_min_sd( i, mm_move_sd( i, i ) )
  result = i.mm_cvtsd_f64


inline float horizontalMinimum( __m256 v )
{
    __m128 i = _mm256_extractf128_ps( v, 1 );
    i = _mm_min_ps( i, _mm256_castps256_ps128( v ) );
    i = _mm_min_ps( i, _mm_movehl_ps( i, i ) );
    i = _mm_min_ss( i, _mm_movehdup_ps( i ) );
    return _mm_cvtss_f32( i );

func hmin*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = ps ) :tt =
  when V is M256i:
    when tt is epi64:
      var i = vec.mm256_extractf128_si256( 1'i32 )
      i = min( i, mm_movehl_ps( i, i ), epi64 )
      i = mm_min_ss( i, mm_movehdup_ps( i ) )
      result = mm_cvtss_si64( i )
        # y  = blend( vec, vec, 1'i32 )
        # m1 = min( vec, y, epi64 )      # m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        # m2 = shuffleLanes( m1, 5'i32 ) # set m2[0] = m1[1], m2[1] = m1[0], etc.
        # r  = min( m1, m2, epi64 )   # all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
      result = r[0, tt]             # extract from idx-0


func hmin*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = epi64 ) :tt =
  when V is M256i:
    when tt is epi64:
      let
        y  = blend( vec, vec, 1 )
        m1 = min( vec, y, tt )      # m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        m2 = shuffleLanes( m1, 5 )  # set m2[0] = m1[1], m2[1] = m1[0], etc.
        r  = min( m1, m2, epi64 )   # all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
      result = r[0, tt]             # extract from idx-0
      # log "y  ", y.as
      # log "m1 ", m1.as
      # log "m2 ", m2.as
      # log "r  ", r.as
]#

proc hmin*[V :M256i]( vec :V, tt :typedesc[SomeSIMD] = epi64, idx :bool = false ) :tt =
  var vmin = vec

  when tt.sizeof == 1: vmin = min( vmin, alignr( vmin, vmin, 1 ), tt )
  when tt.sizeof <= 2: vmin = min( vmin, alignr( vmin, vmin, 2 ), tt )
  when tt.sizeof <= 4: vmin = min( vmin, alignr( vmin, vmin, 4 ), tt )

  vmin = min( vmin, alignr( vmin, vmin, 8 ), tt )
  vmin = min( vmin, blend( vmin, vmin, 0x01, epi128 ), tt )

  if idx:
    let vcmp  = vec.cmpEq( vmin, tt )
    let mask  = vcmp.moveMask()
    let index = tzu32(mask) shr (fastLog2 tt.sizeof)
    #result = tt( index )

    when tt is epi64: result = index.int64
    elif tt is epu64: result = index.uint64
    elif tt is epi32: result = index.int32
    elif tt is epu32: result = index.uint32
    elif tt is epi16: result = index.int16
    elif tt is epu16: result = index.uint16
    elif tt is epi8:  result = index.int8
    elif tt is epu8:  result = index.uint8
  else:
    result = vmin[0, tt]


proc hmin*[V :M256d | M256]( vec :V, tt :typedesc[SomeSIMD] = pd, idx :bool = false ) :tt =
  var vmin = vec
  when tt.sizeof <= 4: vmin = min( vmin, alignr( vmin, vmin, 4 ), tt )

  vmin = min( vmin, alignr( vmin, vmin, 8 ), tt )
  vmin = min( vmin, blend( vmin, vmin, 0x01, epi128 ), tt )
  result = if idx:
      let mask = vec.cmp( vmin, 0'i32 ).moveMask()
      tt( tzu32 mask )
    else:
      vmin[0, tt]


proc hmax*[V :M256 | M256d]( vec :V, tt :typedesc[SomeSIMD] = pd, idx :bool = false ) :tt =
  var vmax = vec

  when tt.sizeof == 4: vmax = max( vmax, alignr( vmax, vmax, 4 ), tt )
  vmax = max( vmax, alignr( vmax, vmax, 8 ), tt )
  vmax = max( vmax, blend( vmax, vmax, 0x01, epi128 ), tt )

  result = if idx:
      let mask = vec.cmp( vmax, 0'i32 ).moveMask()
      tt( tzu32 mask )
    else:
      vmax[0, tt]


proc hmax*[V :M256i]( vec :V, tt :typedesc[SomeSIMD] = epi64, idx :bool = false ) :tt =
  var vmax = vec
  #when tt is epi64:
    # let
    #   y  = blend( vec, vec, 1 )
    #   m1 = max( vec, y, tt )      # m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
    #   m2 = shuffleLanes( m1, 5 )  # set m2[0] = m1[1], m2[1] = m1[0], etc.
    #   r  = max( m1, m2, epi64 )   # all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
    #result = r[0, tt].int64       # extract from idx-0

  when tt.sizeof == 1: vmax = max( vmax, alignr( vmax, vmax, 1 ), tt )
  when tt.sizeof <= 2: vmax = max( vmax, alignr( vmax, vmax, 2 ), tt )
  when tt.sizeof <= 4: vmax = max( vmax, alignr( vmax, vmax, 4 ), tt )

  vmax = max( vmax, alignr( vmax, vmax, 8 ), tt )
  vmax = max( vmax, blend( vmax, vmax, 0x01, epi128 ), tt )

  if idx:
    let vcmp  = vec.cmpEq( vmax, tt )
    let mask  = vcmp.moveMask()
    let index = tzu32(mask) shr (fastLog2 tt.sizeof)
    #result = index
    when tt is epi64: result = index.int64
    elif tt is epu64: result = index.uint64
    elif tt is epi32: result = index.int32
    elif tt is epu32: result = index.uint32
    elif tt is epi16: result = index.int16
    elif tt is epu16: result = index.uint16
    elif tt is epi8:  result = index.int8
    elif tt is epu8:  result = index.uint8
  else:
    result = vmax[0, tt]

  # elif V is M256d:
  #   let
  #     y  = shuffleLanes( vec, vec, 1'i32 )
  #     m1 = max( vec, y, pd )         # m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
  #     m2 = shuffleLanes( m1, 5'i32 ) # set m2[0] = m1[1], m2[1] = m1[0], etc.
  #     r  = max( m1, m2, pd )         # all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
  #   result = r[0, tt].float64        # extract from idx-0
  # elif V is M256:
  #   let
  #     y  = shuffleLanes( vec, vec, 1'i32 )
  #     m1 = max( vec, y, ps )         # m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
  #     m2 = shuffleLanes( m1, 5'i32 ) # set m2[0] = m1[1], m2[1] = m1[0], etc.
  #     r  = max( m1, m2, ps )         # all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
  #   result = r[0, tt].float32        # extract from idx-0

#[
uint8_t _mm256_hmax_index(const __m256i v)
{
    __m256i vmax = v;

    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
    vmax = _mm256_max_epu32(vmax, _mm256_permute2x128_si256(vmax, vmax, 0x01));

    __m256i vcmp = _mm256_cmpeq_epi32(v, vmax);

    uint32_t mask = _mm256_movemask_epi8(vcmp);

    return __builtin_ctz(mask) >> 2;
}
]#

func hmax_idx*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = epi32 ) :int =
  # PERF for epi32/epi64 can be more efficient.
  ## finds horizontal index of max-element in M256i-vector of type `tt`
  result = vec.hmax( tt, true ).int

func hmin_idx*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = epi32 ) :int =
  # PERF for epi32/epi64 can be more efficient.
  ## finds horizontal index of min-element in M256i-vector of type `tt`
  result = vec.hmin( tt, true ).int


#[
  // src ::  https://stackoverflow.com/questions/77317083/how-to-align-rotate-a-256-bit-vector-in-avx2

_m256i rotate2( __m256i v )
{
    // Make another vector with 16-byte pieces flipped
    __m256i flipped = _mm256_permute2x128_si256( v, v, 0x01 );
    // With these two vectors, `vpalignr` can rotate the complete input
    return _mm256_alignr_epi8( v, flipped, 12 );
}
]#
func rotate2( vec :M256i ) :M256i =
  var flipped = mm256_permute2x128_si256( vec, vec, 0x01 )
  result = vec.mm256_alignr_epi8( flipped, 12 )

func rotate*(  vec :var M256i ) =
  vec = vec.mm256_permute4x64_epi64 0b10_01_00_11'u32

func rotate*(  vec :var M128i ) :M128i = vec.mm_alignr_epi8(vec, 4'i32)

func rotateRight*(  vec :var M256i ) = vec.rotate()

func rotateLeft*(  vec :var M256i ) =
  vec = vec.mm256_permute4x64_epi64 0b00_11_10_01'u32

func extract*[ T: SomeInteger ]( vec :M256i, idx :static int32 ) :int =
  when T is int64: result = vec.mm256_extract_epi64( idx ).int
  elif T is int32: result = vec.mm256_extract_epi32( idx ).int
  elif T is int16: result = vec.mm256_extract_epi16( idx ).int
  elif T is int8:  result = vec.mm256_extract_epi8(  idx ).int
  # elif T is float32:  result = vec.mm256_extract_ps(  idx ).int
  # elif T is int8:  result = vec.mm256_extract_epi8(  idx ).int

func `[]`*[V : SomeVec]( vec :V, i :static int, tt :typedesc[SomeSIMD] ) :tt =
  let idx = i.int32
  when V is M256d: vec.mm256_cvtsd_f64()
  elif V is M256:  vec.mm256_cvtss_f32()
  elif V is M256i:
    when tt is epi64:  vec.mm256_extract_epi64( idx )
    elif tt is epu64:  vec.mm256_extract_epi64( idx ).uint64
    elif tt is epi32:  vec.mm256_extract_epi32( idx )
    elif tt is epu32:
      let v = vec.mm256_extract_epi32( idx )
      result = epu32( cast[ptr uint32]( v.addr )[] )
    elif tt is epu16:
      let v = vec.mm256_extract_epi16( idx )
      result = epu16( cast[ptr uint16]( v.addr )[] )
    elif tt is epi16:
      let v = vec.mm256_extract_epi16( idx )
      result = epi16( cast[ptr int16]( v.addr )[] )
    elif tt is epu8 :
      let v = vec.mm256_extract_epi16( idx )
      result = epu8( cast[ptr uint8]( v.addr )[] )
    elif tt is epi8 :
      let v = vec.mm256_extract_epi16( idx )
      result = epi8( cast[ptr int8]( v.addr )[] )
  elif V is M128i:
    when tt is epi64: vec.mm_extract_epi64( idx )
    elif tt is epi32: vec.mm_extract_epi32( idx )
    elif tt is epi16: vec.mm_extract_epi16( idx )
    elif tt is epi8 : vec.mm_extract_epi8(  idx )

func `[]`*[V :SomeVec]( vec :V, i :static SomeInteger ) :SomeInteger|SomeFloat =
  let idx = i.int32
  when V is M256i:
    when i is int64: vec.mm256_extract_epi64( idx ).int64
    elif i is int32:
      let v = vec.mm256_extract_epi32( idx ).uint16
      result = epi32( cast[ptr int32]( v.addr )[] )
    elif i is uint32: vec.mm256_extract_epi32( idx ).uint32
    elif i is int16:
      let v = vec.mm256_extract_epi16( idx ).uint32
      result = epi16( cast[ptr int16]( v.addr )[] )
    elif i is uint16: vec.mm256_extract_epi16( idx ).uint16
    elif i is int8  : vec.mm256_extract_epi8(  idx ).int8
    elif i is uint8 : vec.mm256_extract_epi8(  idx ).uint8
  elif V is M128i:
    when i is epi64: vec.mm_extract_epi64( idx )
    elif i is epi32: vec.mm_extract_epi32( idx )
    elif i is epi16: vec.mm_extract_epi16( idx ).int16
    elif i is epi8 : vec.mm_extract_epi8(  idx ).int8
  elif V is M256d: vec.mm_extract_pd( idx )
  elif V is M256:  vec.mm_extract_ps( idx )


func extractLane*[V :V256]( vec :V, i :static int = 0 ) :V128 =
  let idx = i.int32
  when V is M256i: vec.mm256_extracti128_si256( i )
  elif V is M256d: vec.mm256_extractf128_pd(    i )
  elif V is M256:  vec.mm256_extractf128_ps(    i )
  # vec.mm256_extractf128_si256 idx.int32 # AVX-version of the above

func hiLane*( vec :V256 ) :V128 = vec.extractLane 1
func loLane*( vec :V256 ) :V128 = vec.extractLane 0


func put*[V : SomeVec, T :SomeSIMD]( vec :var V, idx :static int, val :T ) =
  when V is M256i:
    when T is epi64:
      vec = vec.mm256_insert_epi64( val, idx.int32 )

func `[]=`*[V : SomeVec, T :SomeSIMD]( vec :var V, idx :static int32, val :T ) =
  when V is M256i:
    when T is epi64: vec = vec.mm256_insert_epi64( val, idx )
    elif T is epi32: vec = vec.mm256_insert_epi32( val, idx )
    elif T is epi16: vec = vec.mm256_insert_epi16( val, idx )
    elif T is epi8 : vec = vec.mm256_insert_epi8(  val, idx )
  elif V is M256:  vec = vec.mm256_insert_ps(  val,     idx )
  elif V is M256d: vec = vec.mm256_insert_pd(  val,     idx )
  elif V is M128i:
    when T is epi64: vec = vec.mm_insert_epi64( val, idx )
    elif T is epi32: vec = vec.mm_insert_epi32( val, idx )
    elif T is epi16: vec = vec.mm_insert_epi16( val, idx )
    elif T is epi8 : vec = vec.mm_insert_epi8(  val, idx )
  elif V is M128:  vec = vec.mm_insert_ps(  val,     idx )
  elif V is M128d: vec = vec.mm_insert_pd(  val,     idx )


# func extract*[ T: SomeInteger ]( vec :M256i, idx :static int32 ) :int =
#   when T is int64: result = vec.mm256_extract_epi64( idx ).int
#   elif T is int32: result = vec.mm256_extract_epi32( idx ).int
#   elif T is int16: result = vec.mm256_extract_epi16( idx ).int
#   elif T is int8:  result = vec.mm256_extract_epi8(  idx ).int

func loadI*(  loc :pointer )  :M256i     = mm256_load_si256   loc
func uloadI*( loc :pointer )  :M256i     = mm256_loadu_si256  loc

func load128I*(  loc :pointer )  :M128i     = mm_load_si128   loc
func uload128I*( loc :pointer )  :M128i     = mm_loadu_si128  loc
func load128*(   loc :pointer )  :M128      = mm_load_ps   loc
func uload128*(  loc :pointer )  :M128      = mm_loadu_ps  loc
func load128D*(  loc :pointer )  :M128d     = mm_load_pd   loc
func uload128D*( loc :pointer )  :M128d     = mm_loadu_pd  loc

func loadS*(   loc :pointer )  :M256     = mm256_load_ps      loc
func loadD*(   loc :pointer )  :M256d    = mm256_load_pd      loc
func uloadS*(  loc :pointer )  :M256     = mm256_loadu_ps     loc
func uloadD*(  loc :pointer )  :M256d    = mm256_loadu_pd     loc

func stream*(  vec :M256i, loc :pointer ) = mm256_stream_si256( loc, vec )
func storeI*(  vec :M256i, loc :pointer ) = mm256_store_si256(  loc, vec )
func ustoreI*( vec :M256i, loc :pointer ) = mm256_storeu_si256(  loc, vec )

func storeI*(  vec :M128,  loc :pointer ) = mm_store_ps(    loc, vec )
func storeI*(  vec :M128d, loc :pointer ) = mm_store_pd(    loc, vec )
func storeI*(  vec :M128i, loc :pointer ) = mm_store_si128( loc, vec )


func fill*( hi,lo :V128 ) :V256 =
  when hi is M128i: mm256_set_m128i( hi, lo )
  elif hi is M128d: mm256_set_m128d( hi, lo )
  elif hi is M128:  mm256_set_m128(  hi, lo )

func fill*[V :V256]( vec :V; hi,lo :V128 ) :V =
  when V is M256i: vec.mm256_set_m128i( hi, lo )
  elif V is M256d: vec.mm256_set_m128d( hi, lo )
  elif V is M256:  vec.mm256_set_m128(  hi, lo )


func fill*[V :V128]( srcV :V, tt :typedesc[SomeSIMD] = epi64 ) :V128 =
  # all cost 3/1
  when V is M128i:
    when tt is epi64:   mm_broadcastq_epi64  srcV
    elif tt is epi32:   mm_broadcastd_epi32  srcV
    elif tt is epi16:   mm_broadcastw_epi16  srcV
    elif tt is epi8:    mm_broadcastb_epi8   srcV
    #elif tt is epi128:  mm_broadcastsi128    srcV # ! cost 7/1 !
  elif V is M128:     mm_broadcastss_ps    srcV
  elif V is M128d:    mm_broadcastsd_pd    srcV

func fill*( val :SomeNumber, tt :typedesc[SomeSIMD] = epi64 ) :SomeVec =
  when val is epi64: result = mm256_set1_epi64x( val )
  when val is epi32: result = mm256_set1_epi32(  val )
  when val is epi16: result = mm256_set1_epi16(  val )
  when val is  epi8: result = mm256_set1_epi8(   val )
  when val is    pd: result = mm256_set1_pd(     val )
  when val is    ps: result = mm256_set1_ps(     val )

func fill128*[T :SomeSIMD]( data :openArray[T], len :int = -1 ) :V128 =
  let loc = cast[ ptr UArr[T] ]( data[0].addr )
  when T is epi: result = uload128I loc
  elif T is ps:  result = uload128  loc
  elif T is pd:  result = uload128D loc

#[
  var big :bool
  if len > 0:
    big = (len div sizeof(T)) * 8 > 128
  elif data.len > 0:
    big = (data.len div sizeof(T)) * 8 > 128

  let arr = cast[ ptr UArr[T] ]( data[0].addr )
  result = if big:
      when T is epi: uloadI arr.addr
      elif T is ps:  uloadS arr.addr
      elif T is pd:  uloadD arr.addr
    else:
      when T is epi: uload128I arr.addr
      elif T is ps:  uload128  arr.addr
      elif T is pd:  uload128D arr.addr
  ]#

template zeroes*( tt :typedesc[SomeSIMD] = epi64 ) :SomeVec =
  when tt is epi:  mm256_setzero_si256()
  elif tt is ps:   mm256_setzero_ps()
  elif tt is pd:   mm256_setzero_pd()

func newM128*[T :SomeSIMD]() :V128 =
  when T is epi: zeroes128 epi8
  elif T is ps:  zeroes128 ps
  elif T is pd:  zeroes128 pd


template zeroes128*( tt :typedesc[SomeSIMD] = epi64 ) :SomeVec =
  when tt is epi:   mm_setzero_si128()
  elif tt is ps:    mm_setzero_ps()
  elif tt is pd:    mm_setzero_pd()

template undefined*( tt :typedesc[SomeSIMD] = epi64 ) :SomeVec =
  when tt is epi:   mm256_undefined_si256()
  elif tt is ps:    mm256_undefined_ps()
  elif tt is pd:    mm256_undefined_pd()


proc has*( vec :M256i, needle :SomeInteger ) :int =

  #let resultV = vec.mm256_cmpeq_epi64( fill( needle.int64 ) )
  var mask :int32
  when needle is int  :
    let resultV = vec.mm256_cmpeq_epi64 fill( needle )
    mask = mm256_movemask_pd( resultV.asM256d )
  elif needle is int32:
    let resultV = vec.mm256_cmpeq_epi32 fill(needle)
    mask = mm256_movemask_ps( resultV.asM256 )
  else:
    let resultV = vec.mm256_cmpeq_epi8 fill(needle)
    mask = mm256_movemask_epi8( resultV )

  # key was not present in vector -> done
  result = if mask != 0: mask.firstSetBit - 1 else: -1


proc dump*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = epi64 ) :string =
  when V is M256i:
    when tt is epi64:
      var arr :array[ 4, int64 ]
      vec.storeI arr[0].addr
      result = "i64-| " & ($arr)[ 1 ..< ($arr).len-1 ]  & fmt" |{$V}|"
    elif tt is epu64:
      var arr :array[ 4, uint64 ]
      vec.storeI arr[0].addr
      result = "u64-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epi32:
      var arr :array[ 8, int32 ]
      vec.storeI arr[0].addr
      result = "i32-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epu32:
      var arr :array[ 8, uint32 ]
      vec.storeI arr[0].addr
      result = "u32-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epi16:
      var arr :array[ 16, int16 ]
      vec.storeI arr[0].addr
      result = "i16-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epu16:
      var arr :array[ 16, uint16 ]
      vec.storeI arr[0].addr
      result = "u16-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epi8:
      var arr :array[ 32, int8 ]
      vec.storeI arr[0].addr
      result = "i8-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif tt is epu8:
      var arr :array[ 32, uint8 ]
      vec.storeI arr[0].addr
      result = "u8-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
  elif (V is M256):
    var arr :array[  8, float32 ]
    vec.asM256i.storeI arr[0].addr
    result = "f32-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
  elif (V is M256d):
    var arr :array[  4, float64 ]
    vec.asM256i.storeI arr[0].addr
    result = "f64-| " &  ($arr)[ 1 ..< ($arr).len-1 ]  & fmt" |{$V}|"
  result = result.replace( ",", "")

template `as`*( vec :V256, tt :typedesc[SomeSIMD] = epi64 ) :string = vec.dump tt

proc dump*[V :V128]( vec :V, tt :typedesc[SomeSIMD] = epi64 ) :string =
  when V is M128i:
    when (tt is epi64):
      var arr :array[ 2, tt ]
      vec.storeI arr[0].addr
      result = "i64-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif (tt is epi32):
      var arr :array[ 4, tt ]
      vec.storeI arr[0].addr
      result = "i32-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif (tt is epi16):
      var arr :array[ 8, tt ]
      vec.storeI arr[0].addr
      result = "i16-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
    elif (tt is epi8) or (tt is epu8):
      var arr :array[ 16, byte ]
      vec.storeI arr[0].addr
      result = "i8-| " & ($arr)[ 1 ..< ($arr).len-1 ] & fmt" |{$V}|"
  elif (V is M128):
    var arr :array[ 4, float32 ]
    vec.storeI arr[0].addr
    result = "f64-| " &  ($arr)[ 1 ..< ($arr).len-1 ]  & fmt" |{$V}|"
  elif (V is M128d):
    var arr :array[ 2, float64 ]
    vec.storeI arr[0].addr
    result = "f32-| " &  ($arr)[ 1 ..< ($arr).len-1 ]  & fmt" |{$V}|"

  result = result.replace( ",", "")

proc `as`*( vec :V128, tt :typedesc[SomeSIMD] = epi64 ) :string = vec.dump tt

proc `()`*( vec :SomeVec, tt :typedesc[SomeSIMD] ) :string = vec.as tt


# func allEq*[V :V256]( vec :V, tt :typedesc[SomeSIMD] = epi8 ) :bool =
#   # src :: http://0x80.pl/notesen/2021-02-02-all-bytes-in-reg-are-equal.html
#   const allEqualMask = 0xffffffff'u32
#   let
#     lowerLane          = vec.asM128i
#     tmp                = lowerLane.fill( tt ).asM256i # mm_broadcast_<tt>
#     populated1stSlot   = blend( tmp, tmp, 0b00_00_00_00, epi128 )
#     eq                 = cmpEq( vec.asM256i, populated1stSlot, tt )
#     mask               = eq.moveMask()
#
#   result = mask == allEqualMask



# stuff that Intel forgot

func mm256_shuffle_epi32*( a, b :M256i, imm8 :static int32 ) :M256i =
  mm256_castps_si256(
    mm256_shuffle_ps(
      mm256_castsi256_ps(a), mm256_castsi256_ps(b), imm8
    )
  )
func mm256_shuffle_epi64*( a, b :M256i, imm8 :static int32 ) :M256i =
  mm256_castpd_si256(
    mm256_shuffle_pd(
      mm256_castsi256_pd(a), mm256_castsi256_pd(b), imm8
    )
  )
func mm_shuffle_epi32*( a, b :M128i, imm8 :static int32 ) :M128i =
  mm_castps_si128(
    mm_shuffle_ps( mm_castsi128_ps(a), mm_castsi128_ps(b), imm8 )
  )
func mm_shuffle_epi64*( a, b :M128i, imm8 :static int32 ) :M128i =
  mm_castpd_si128(
    mm_shuffle_pd( mm_castsi128_pd(a), mm_castsi128_pd(b), imm8 )
  )


# include simdRecipies/mula

##[
  https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av
]##
# func hsum_epi32_avx*( vec :M128i ) :uint32 =
#   let
#     hi64  = mm_unpackhi_epi64(vec, vec)  # 3-operand non-destructive AVX lets us save a byte without needing a movdqa
#     sum64 = mm_add_epi32( hi64, vec)
#     hi32  = mm_shuffle_epi32(sum64, MM_SHUFFLE(2, 3, 0, 1))  # Swap the low two elements
#     sum32 = mm_add_epi32( sum64, hi32 )
#   result = mm_cvtsi128_si32 sum32       # movd

# func hsum_8x32*( vec :M256i | M256 ) :uint32 =
#   # only needs AVX2
#   when vec is M256i:
#     let sum128 = mm_add_epi32(
#       mm256_castsi256_si128(  vec   ),
#       mm256_extracti128_si256(vec, 1)
#     ) # silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
#     result = hsum_epi32_avx( sum128 )
#   elif vec is M256:
#     let sum128 = mm_add_ps(
#       mm256_castps_ps128(  vec   ),
#       mm256_extractf128_ps( vec, 1)
#     )
#     result = hsum_epi32_avx( sum128.asM128i )

#[
static inline __m256i _mm256_alignr_epi8(const __m256i v0, const __m256i v1, const int n)
{
  if (n < 16)
  {
    __m128i v0h = _mm256_extractf128_si256(v0, 0);
    __m128i v0l = _mm256_extractf128_si256(v0, 1);
    __m128i v1h = _mm256_extractf128_si256(v1, 0);
    __m128i vouth = _mm_alignr_epi8(v0l, v0h, n);
    __m128i voutl = _mm_alignr_epi8(v1h, v0l, n);
    __m256i vout = _mm256_set_m128i(voutl, vouth);
    return vout;
  }
  else
  {
    __m128i v0h = _mm256_extractf128_si256(v0, 1);
    __m128i v0l = _mm256_extractf128_si256(v1, 0);
    __m128i v1h = _mm256_extractf128_si256(v1, 1);
    __m128i vouth = _mm_alignr_epi8(v0l, v0h, n - 16);
    __m128i voutl = _mm_alignr_epi8(v1h, v0l, n - 16);
    __m256i vout = _mm256_set_m128i(voutl, vouth);
    return vout;
  }
}
 ]#