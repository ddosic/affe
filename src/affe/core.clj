;;inspired by and reusing some code from Carin Meier K9 demonstration project

(ns affe.core
    (:require 
              [affe.protocols :refer :all]
              [affe.trainer :refer :all]
              [affe.cpu.engine :refer :all]
              [affe.opencl.engine :refer :all]
              [uncomplicate.clojurecl.core  :refer [*context* *command-queue* with-default]]

              [uncomplicate.neanderthal
             [core :refer [transfer]]
             [opencl :refer [with-default-engine]]]))

(defn generate-weights [engine size-in size-hidden num-hidden size-out]
           (vec (concat
                    [(.gen-strengths engine size-in size-hidden)]
                    (->>
                     [(.gen-strengths engine  size-hidden size-hidden)]
                      (repeat (dec num-hidden))
                      (apply concat))
                     [(.gen-strengths engine  size-hidden size-out)])))

(defn construct-network
  ([affe-engine size-in size-hidden size-out]
  "construct a three layer neural network"
  (construct-network affe-engine size-in size-hidden 1 size-out))
  ([affe-engine size-in size-hidden num-hidden size-out]
  "construct a N layer neural network"
  (generate-weights affe-engine size-in size-hidden num-hidden size-out)))

