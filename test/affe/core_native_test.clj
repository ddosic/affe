(ns affe.core-native-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.network :refer :all]
            [affe.cpu.engine :refer :all]
            [affe.trainer :refer :all]
            [affe.gui :refer :all]))

(facts
 "Hopefully no crash here."
 (def visitor (->NeuralNetworkVisitor (native-affe-engine)))
 (def testNet (construct-network (native-affe-engine) 3 3 3))
 (def trained (train-epochs visitor testNet 10000 [[[1 2 3][-1 5 1]][[4 5 3][1 5 3]][[0 2 0][-1 5 -1]][[9 8 0][1 -5 1]]] 0.001))
 (show (network-graph visitor trained))
 )
