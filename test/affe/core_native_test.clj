(ns affe.core-native-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.network :refer :all]
            [affe.cpu.engine :refer :all]
            [affe.trainer :refer :all]
            [affe.gui :refer :all]
            [uncomplicate.clojurecl.core  :refer [*context* *command-queue* with-default]]
            [uncomplicate.neanderthal
             [core :refer [transfer]]
             [opencl :refer [with-default-engine]]]
            ))
(facts
 "Hopefully no crash here."
 (def naf (native-affe-engine))
 (def visitor (->NeuralNetworkVisitor naf))
 (def testNet (construct-network naf 4 4 4))
 (def trained (train visitor testNet 10 [[[(vec (range 4))][[1 -1 1 0]]][[(vec (range 4))][[1 -1 1 0]]]] 0.1))
 (show (network-graph visitor [4 4 4] trained))
 )
