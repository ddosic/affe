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
             [opencl :refer [with-default-engine]]]
            ))
(with-default
  (with-default-engine
    (facts
       "Hopefully no crash here."
       (def engine (cl-affe-engine *context* *command-queue*))
       (def visitor (->NeuralNetworkVisitor engine))
       (def testNet (construct-network engine 3 3 3))
       ;(def testNeta (.ff visitor testNet [1 2 3]))
       (def testNeta (train-network visitor testNet (.wrap-input engine [1 2 3]) (.wrap-input engine [0 0 1]) 0.1))
              (def host-testNeta [(vec (map transfer (.get-layers visitor testNeta)))
                         (vec (map transfer (.get-weights visitor testNeta)))])
       (println "gpua" host-testNeta)

       (def trained (train-epochs visitor testNet 100 [[[1 2 3][-1 5 1]][[4 5 3][1 5 3]][[0 2 0][-1 5 -1]][[9 8 0][1 -5 1]]] 0.001))
       (def host-trained [(vec (map transfer (.get-layers visitor trained)))
                         (vec (map transfer (.get-weights visitor trained)))])
       (show (network-graph visitor host-trained)))))
