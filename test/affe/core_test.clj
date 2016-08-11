(ns affe.core-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]))



(facts
 "Hopefully no crash here."
 (def testNet (construct-network 3 3 3))
 (println (ff  [1 2 3] testNet))
 (def trainedNet (train-epochs  10000 testNet [[[1 2 3][2 5 -1]]] 0.001))
 (println (ff  [1 2 3] trainedNet))
 )
 
