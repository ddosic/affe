(ns affe.trainer
     (:require [affe.protocols :refer :all]
               [uncomplicate.commons.core :refer [let-release release]]))
   
   (defn train-network [visitor network input target learning-rate]
     "train network with one set of target data"
     (.update-weights visitor (.feed-forward visitor network input ) network target learning-rate))
   
   (defn train-data [visitor network data learning-rate]
       (if-let [[input target] (first data)]
               (recur
                visitor (train-network visitor network input target learning-rate)
                (rest data)
                learning-rate)
               network))
              
   
   (defn train-epochs [visitor network n training-data learning-rate]
     (println "round " n)
       (if (zero? n)
           network
           (recur visitor
                 (train-data visitor network training-data learning-rate)
                 (dec n)
                  training-data
                  learning-rate)))
   
   (defn train [visitor network n training-data learning-rate]
     (let-release [wrap-training-data (map(fn [[input target]] [(.prepare-input visitor input) 
                                                                (.prepare-input visitor target)] ) 
                                          training-data)]
       (train-epochs visitor network n wrap-training-data learning-rate)
       )
     )