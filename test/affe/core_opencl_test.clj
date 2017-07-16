(ns affe.core-opencl-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.protocols :refer :all]
            [affe.network :refer :all]
            [affe.opencl.engine :refer :all]
            [affe.trainer :refer :all]
            [affe.gui :refer :all]
            [uncomplicate.clojurecl.core  :refer [*context* *command-queue* with-default finish!]]
            [uncomplicate.neanderthal
             [core :refer [transfer]]
             [opencl :refer [with-engine opencl-double]]]
            ))
(with-default
  (with-engine opencl-double *command-queue*
    (facts
       "Hopefully no crash here."
       (def engine (cl-affe-engine *context* *command-queue*))
       (def visitor (->NeuralNetworkVisitor engine))
       (def testNet (construct-network engine 4 4 4))
       (def trained (train visitor testNet 10 [[[(vec (range 4))][[1 -1 1 0]]]] 0.1))
       (def host-trained (vec (map transfer  trained)))
       (show (network-graph visitor [4 4 4] host-trained)))))

