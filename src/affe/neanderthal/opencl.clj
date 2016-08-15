(ns affe.neanderthal.opencl

(:require [affe.core :refer :all]
          [uncomplicate.commons.core :refer [with-release]]
          [uncomplicate.clojurecl.core :refer [with-default finish!]]
          [uncomplicate.neanderthal
             [core :refer :all]
             [native :refer :all]
             [opencl :refer :all]]))

(defn gen-strengths [from to]
  "generate random strengths for layer"
  (let [l (* from to )]
    (clge  to from( vec (repeat l 0.01)))))

(defn train-data [network data learning-rate]
  "train network with a set of data in the form of [[input1 target1] [input2 target2]]"
  (if-let [[input target] (first data)]
    (let [gpu-input-block (clv input) 
                    gpu-target-block (clv target)]
    (recur
     (train-network network gpu-input-block gpu-target-block learning-rate)
     (rest data)
     learning-rate))
    network))

(defn train-epochs [n network training-data learning-rate]
  "train repeatedly n times over the same tranining data in the form of [[input1 target1] [input2 target2]]  "
  (if (zero? n)
    network
    (recur (dec n)
           (train-data network training-data learning-rate)
           training-data
           learning-rate)))

(defn construct-network
  ([num-in num-hidden num-out]
  "construct a three layer neural network"
  (construct-network num-in num-hidden 0 num-out))
  ([size-in size-hidden num-hidden size-out]
    "construct a N layer neural network"
    [(vec (concat
                    [(clv (repeat size-in 0))]
                    (->>
                     [(clv (repeat size-hidden 0))]
                     (repeat  num-hidden)
                     (apply concat))
                     [(clv (repeat size-out 0))]))
     (vec (concat
                    [(gen-strengths size-in size-hidden)]
                    (->>
                      [(gen-strengths size-hidden size-hidden)]
                      (repeat (dec num-hidden))
                      (apply concat))
                    [(gen-strengths size-hidden size-out)]))
     ]))

(defn ff [input network]
  "Feed forward and return output neurons"
  (with-release [ gpu-input (clv input)]
    (last (get-layers (feed-forward gpu-input network)))))

(defn release-resources [network]
  "Releases all memory on device reserved used by this network"
  (with-release [weights (get-weights network)
                 neurons (get-layers network)]))
