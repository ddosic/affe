(ns affe.core-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]))



(facts
 "Hopefully no crash here."
 (def testNet (construct-network 3 3 3))
 (train-epochs  1000 testNet [[[1 2 3][2 5 6]]] 0.01)
 )
 
