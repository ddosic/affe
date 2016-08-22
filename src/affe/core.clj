(ns affe.core
    (:require 
              [affe.protocols :refer :all]
              [affe.trainer :refer :all]
              [affe.cpu.engine :refer :all]
              [affe.network :refer :all]))

(def ^{:dynamic true
       :doc "Dynamic var for binding the default network engine."}
  *network-engine*)

(defn generate-layers [engine size-in size-hidden num-hidden size-out]
      (vec (concat
                    [(.gen-layer engine size-in)]
                    (->>
                     [(.gen-layer engine size-hidden)]
                     (repeat  num-hidden)
                     (apply concat))
                     [(.gen-layer engine size-out)])))

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
  [(generate-layers affe-engine size-in size-hidden num-hidden size-out)
   (generate-weights affe-engine size-in size-hidden num-hidden size-out)]))

