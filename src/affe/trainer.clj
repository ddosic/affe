(ns affe.trainer
  (:require [affe.protocols :refer :all]))

(defn train-network [visitor network input target learning-rate]
  "train network with one set of target data"
  (.update-weights visitor (.feed-forward visitor network input ) target learning-rate))

(defn train-data [visitor network data learning-rate]
    (if-let [[input target] (first data)]
            (let [input-block (.prepare-input visitor input) 
                  target-block (.prepare-input visitor target)]
            (recur
             visitor (train-network visitor network input-block target-block learning-rate)
             (rest data)
             learning-rate))
            network)
           )

(defn train-epochs [visitor network n training-data learning-rate]
  (println "round " n)
    (if (zero? n)
        network
        (recur visitor
                (train-data visitor network training-data learning-rate)
                (dec n)
               training-data
               learning-rate)))
