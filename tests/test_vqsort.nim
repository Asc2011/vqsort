import std/[monotimes, random, algorithm, strformat ]
from std/stats import mean

from std/algorithm import sort
from vqsort import quicksort

randomize()
proc ns*() :int64 = monotimes.getMonoTime().ticks

proc mkResults( algo :string, runs :var seq[int] ) :float =
  runs.sort()
  let (best, worst) = ( runs[0], runs[^1] )
  runs = runs[ 1 .. ^2 ]
  #echo runs.len, "x", algo, " (", best, ")-", runs, "-(", worst, ")"
  result = stats.mean runs

proc testVQS( cap :int, turns :int = 12 ) :float =
  var data :seq[int32]
  data.setLen cap
  for i in 0 ..< cap: data[i] = rand(2*cap).int32
  var runs :seq[int]
  for r in 0 ..< turns :
    let t0 = ns()
    quicksort( data[0].addr, cap )
    runs.add (ns() - t0).int
    data.shuffle

  return "VQS".mkResults runs

proc testStd( cap :int, turns :int = 12 ) :float =
  var data :seq[int32]
  data.setLen cap
  for i in 0 ..< cap: data[i] = rand(2*cap).int32
  var runs :seq[int]
  for r in 0 ..< turns:
    let t0 = ns()
    data.sort()
    runs.add (ns() - t0).int
    shuffle data

  return "Std".mkResults runs

# ======================= main ==================================

when isMainModule:

  echo "testing sizes from 0.5 .. 4_000-KB"

  for cap in [ 500, 1_000, 10_000, 50_000, 250_000, 500_000, 1_000_000, 4_000_000 ]:
    #
    # take 12-samples
    # drop the best- and the worst-sample
    # return the mean-of-remaining 10-samples
    #
    let
      vqsMean = (testVQS cap) / 1000
      stdMean = (testStd cap) / 1000
      theX    = stdMean / vqsMean

      p0 = fmt"{(cap/1000):.1f}-KB "
      p1 = fmt"std/sort {stdMean:.2f}"
      p2 = fmt" | ~{theX:.1f}X | "
      p3 = fmt"{vqsMean:.2f} vectorized Quicksort"

    echo "\n",p0,"\t",p1,p2,p3