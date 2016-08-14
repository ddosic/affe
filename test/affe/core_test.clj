(ns affe.core-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.gui :refer :all]))

(facts
 "Hopefully no crash here."
 (def testNet (construct-network 3 3 3))
 (println (ff  [1 2 3] testNet))
 (def trainedNet (train-epochs  10000 testNet [[[1 2 3][-1 5 1]][[4 5 3][1 5 3]][[0 2 0][-1 5 -1]][[9 8 0][1 -5 1]]] 0.001))
 (println trainedNet)
 (show (network-graph trainedNet))
 )
